"""
Fine-tuning DiT with minimax criteria.
"""
import sys
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.utils import save_image

import numpy as np
from collections import OrderedDict, defaultdict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from data_2 import load_resized_data
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import tqdm
import numpy as np
import torch
import PIL.Image
import torch.nn.functional as F
import dnnlib
import pickle
from training.networks import SongUNet
from training.loss import EDMLoss

from typing import Dict

import torch.optim as optim


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logger = logging.getLogger(__name__)
        logger.propagate = False
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        if logging_dir:
            fh = logging.FileHandler(f'{logging_dir}/logs.txt')
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def cosine_similarity(ta, tb):
    bs1, bs2 = ta.shape[0], tb.shape[0]
    frac_up = torch.matmul(ta, tb.T)
    frac_down = torch.norm(ta, dim=-1).view(bs1, 1).repeat(1, bs2) * \
                torch.norm(tb, dim=-1).view(1, bs2).repeat(bs1, 1)
    return frac_up / frac_down


def mark_difffit_trainable(model, is_bitfit=False):
    """
    Mark the parameters that require updating by difffit.
    """
    if is_bitfit:
        trainable_names = ['bias']
    else:
        trainable_names = ["bias", "norm", "gamma", "y_embed"]

    for par_name, par_tensor in model.named_parameters():
        par_tensor.requires_grad = any([kw in par_name for kw in trainable_names])
    return model

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Fine-tune a DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    print(f"World size: {dist.get_world_size()}")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        if args.tag:
            experiment_dir += f"-{args.tag}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    # CIFAR-10 model:
    # Unpickle
    with dnnlib.util.open_url(args.pkl) as f:
        pre_trained = pickle.load(f)
    # model = pre_trained['ema'].to(device)
    # torch.save(model.model.state_dict(), 'cifar_ckpt/edm_checkpoint.pth')
    # loss_fn = pre_trained['loss_fn'] # EDM loss function
    augment_pipe = pre_trained['augment_pipe']
    # dataset_kwargs = pre_trained['dataset_kwargs']
    
    # Initialize model
    # and load edm model from .pth
    model = SongUNet(img_resolution=args.size,
                     in_channels=3,
                     out_channels=3,
                     label_dim=args.nclass,
                     augment_dim=9,
                     embedding_type='fourier',
                     encoder_type='residual',
                     channel_mult_noise=2,
                     resample_filter=[1,3,3,1],
                     channel_mult=[2,2,2]).to(device)
    model.load_state_dict(torch.load(args.ckpt))
    loss_fn = EDMLoss()
    

    print(f"Loaded model from {args.ckpt}")
        
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-3 in the Difffit paper):
    model = mark_difffit_trainable(model)
    model = DDP(model.to(device), device_ids=[rank])
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in params_to_optimize)
    print(f"Number of Trainable Parameters: {total_params * 1.e-6:.2f} M")
    #opt = torch.optim.AdamW(params_to_optimize, lr=1e-5, weight_decay=0)

    batch_size = args.global_batch_size
    # Setup optimizer.
    optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=1e-5, betas=[0.9,0.999], eps=1e-8)
    augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', p=1)
    print('Setting up loss function and optimizer...')
    opt = dnnlib.util.construct_class_by_name(params=model.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    #augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe

    # Setup data:
    #dataset = ImageFolder(args.data_path, transform=transform, nclass=args.nclass,
    #                      ipc=args.finetune_ipc, spec=args.spec, phase=args.phase,
    #                      seed=0, return_origin=True)
    dataset, _ = load_resized_data(args)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss, running_loss_pos, running_loss_neg = 0, 0, 0
    start_time = time()
    real_memory = defaultdict(list)
    pseudo_memory = defaultdict(list)

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)      # make sure to shuffle the data differently each epoch
        logger.info(f"Beginning epoch {epoch}...")

        if args.dataset == 'cifar10':
            for x, y in loader:
                ry = y.numpy()
                x = x.to(device)
                y = y.to(device)

                class_labels = F.one_hot(y, args.nclass).float()

                loss = loss_fn(net=model, images=x, labels=class_labels, augment_pipe=augment_pipe)
                #loss.sum().mul(1 / args.global_batch_size % dist.get_world_size())

                print('x size 0:', x.size(0))
                noise_labels =  torch.rand(x.size(0), device=device) * (model.sigma_max - model.sigma_min) + model.sigma_min
                # Sample from the diffusion.
                sampled = model(x, noise_labels, class_labels)
                pseudo_embeddings = sampled
                        
                # Calculate minimax criteria
                pos_match_loss = torch.tensor(0.).to(device)
                neg_match_loss = torch.tensor(0.).to(device)
                if args.condense:
                    ry_set = set(ry)
                    num_y = len(ry_set)
                    for c in ry_set:
                        if len(pseudo_memory[c]):
                            pos_embeddings = torch.cat(real_memory[c]).flatten(start_dim=1)
                            neg_embeddings = torch.cat(pseudo_memory[c]).flatten(start_dim=1)
                            # Representativeness constraint
                            pos_feat_sim = 1 - cosine_similarity(
                                pseudo_embeddings[ry == c].flatten(start_dim=1), pos_embeddings
                            ).min()
                            # Diversity constraint
                            neg_feat_sim = cosine_similarity(
                                pseudo_embeddings[ry == c].flatten(start_dim=1), neg_embeddings
                            ).max()
                            pos_match_loss += pos_feat_sim * args.lambda_pos / num_y
                            neg_match_loss += neg_feat_sim * args.lambda_neg / num_y
                            #logger.info(f'pos_match_loss: {pos_match_loss}, neg_match_loss: {neg_match_loss}')
                        # Update the auxiliary memories
                        real_memory[c].extend(x[ry == c].detach().split(1))
                        pseudo_memory[c].extend(pseudo_embeddings[ry == c].detach().split(1))
                        while len(real_memory[c]) > args.memory_size:
                            real_memory[c].pop(0)
                        while len(pseudo_memory[c]) > args.memory_size:
                            pseudo_memory[c].pop(0)

                    all_loss = loss + pos_match_loss + neg_match_loss
                else:
                    all_loss = loss

                opt.zero_grad()
                all_loss.backward()
                opt.step()

                update_ema(ema, model.module)

                # Log loss values:
                running_loss += loss.item()
                if pos_match_loss or neg_match_loss:
                    running_loss_pos += pos_match_loss.item()
                    running_loss_neg += neg_match_loss.item()
                log_steps += 1
                train_steps += 1
                if train_steps % args.log_every == 0:
                    # Measure training speed:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    # Reduce loss history over all processes:
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    avg_loss_pos = torch.tensor(running_loss_pos / log_steps, device=device)
                    avg_loss_neg = torch.tensor(running_loss_neg / log_steps, device=device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(avg_loss_pos, op=dist.ReduceOp.SUM)
                    dist.all_reduce(avg_loss_neg, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()
                    avg_loss_pos = avg_loss_pos.item() / dist.get_world_size()
                    avg_loss_neg = avg_loss_neg.item() / dist.get_world_size()
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f} {avg_loss_pos:.4f} {avg_loss_neg:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    # Reset monitoring variables:
                    running_loss = 0
                    running_loss_pos = 0
                    running_loss_neg = 0
                    log_steps = 0
                    start_time = time()
                    

                # Save DiT checkpoint:
                if train_steps % args.ckpt_every == 0 and train_steps > 0:
                    if rank == 0:
                        checkpoint = {
                            "model": model.module.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "args": args
                        }
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    dist.barrier()


    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--model", type=str, default="DiT-XL/2")
    parser.add_argument("--size", type=int, choices=[32, 256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000, help='the class number for the total dataset')
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--nclass", type=int, default=10, help='the class number for distillation training')
    parser.add_argument("--finetune-ipc", type=int, default=1000, help='the number of samples participating in the fine-tuning')
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--condense", action="store_true", default=False, help='whether conduct distillation')
    parser.add_argument("--spec", type=str, default='none', help='specific subset for distillation')
    parser.add_argument('--lambda-pos', default=0.002, type=float, help='weight for representativeness constraint')
    parser.add_argument('--lambda-neg', default=0.008, type=float, help='weight for diversity constraint')
    parser.add_argument("--memory-size", type=int, default=64, help='the memory size')
    parser.add_argument("--phase", type=int, default=0, help='the phase number for generating large datasets')
    parser.add_argument("--dataset", type=str, default='cifar10', help='the dataset name')
    parser.add_argument("--data-dir", type=str, default='./datasets', help='the directory to store the dataset')
    parser.add_argument("--download", action='store_true', default=False, help='whether download the dataset')
    #parser.add_argument("--batch-size", type=int, default=64, help='batch size for training')
    parser.add_argument("--pkl", type=str, default=None, help='the pkl file for edm')
    args = parser.parse_args()
    main(args)
