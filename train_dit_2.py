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
from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import tqdm
import numpy as np
import torch
import PIL.Image
import torch.nn.functional as F
import dnnlib
import pickle

from diffusers import DDPMScheduler, UNet2DModel


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
    print(f'Loading network from "{args.ckpt}"...')
    with dnnlib.util.open_url(args.ckpt) as f:
        model = pickle.load(f)['ema'].to(device)

    # load model and scheduler
    #scheduler = DDPMScheduler.from_pretrained('google/ddpm-cifar10-32')
    #model = UNet2DModel.from_pretrained('google/ddpm-cifar10-32', use_safetensors=True)
    # Set class_embed_type
    #model.config.class_embed_type = 'identity' # uses the class labels directly
    #model.num_class_embeds = 10
    #model.to(device)
    print('model config:', model)

    '''
    model = create_model(image_size=args.size,
                     num_channels=128,
                     num_res_blocks=3,
                     learn_sigma=True,
                     class_cond=False,
                     use_checkpoint=False,
                     dropout=0.3,
                     attention_resolutions="18, 6",
                     num_heads=4,
                     num_heads_upsample=-1,
                     num_classes=args.num_classes,
                     use_scale_shift_norm=False,
                     )
    '''
    
    '''
    DiT-model:
    assert args.size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
    )
    
    # Load pretrained model:
    if args.ckpt != None:
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path)
        model.load_state_dict(state_dict, strict=False)
'''
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-3 in the Difffit paper):
    model = mark_difffit_trainable(model)
    model = DDP(model.to(device), device_ids=[rank])
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in params_to_optimize)
    print(f"Number of Trainable Parameters: {total_params * 1.e-6:.2f} M")
    opt = torch.optim.AdamW(params_to_optimize, lr=1e-3, weight_decay=0)

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
        if args.dataset == 'imagenet':
            for x, ry, y in loader:
                ry = ry.numpy()
                x = x.to(device)
                y = y.to(device)
                with torch.no_grad():    # disable gradient computation for the following block
                    # Map input images to latent space + normalize latents:
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                model_kwargs = dict(y=y)   # can be used for conditioning, dict of extra keyword arguments
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                pseudo_embeddings = loss_dict["output"]
                loss = loss_dict["loss"].mean()

                # Calculate minimax criteria
                pos_match_loss = torch.tensor(0.).to(device)
                neg_match_loss = torch.tensor(0.).to(device)
                if args.condense:
                    ry_set = set(ry)
                    num_ry = len(ry_set)
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
                            pos_match_loss += pos_feat_sim * args.lambda_pos / num_ry
                            neg_match_loss += neg_feat_sim * args.lambda_neg / num_ry

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
        if args.dataset == 'cifar10':
            for x, y in loader:
                ry = y.numpy()
                x = x.to(device)
                y = y.to(device)
                #if args.dataset == 'cifar10': # map cifar10 to corresponding imagenet classes
                #    mapping = {0: 404, 1: 436, 2: 94, 3: 281, 4: 352, 5: 207, 6: 32, 7: 339, 8: 510, 9: 864}
                #    ry = np.array([mapping[i] for i in ry])
                #with torch.no_grad():
                    # Map input images to latent space + normalize latents:
                    #x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                    # Upsample the latent code
                    # x = torch.nn.functional.interpolate(x, size=(latent_size, latent_size), mode='bilinear', align_corners=False)
                # Sample noise add to input images
                noise = torch.randn(x.shape).to(device)

                # Sample a random timestep for each image
                t = torch.randint(0, scheduler.num_train_timesteps, (x.shape[0],), device=device)

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = scheduler.add_noise(x, noise, t)

                # Get the model prediction for the noise
                noise_pred = model(noisy_images, t, y, return_dict=False)[0]

                # Compare the prediction with the actual noise:
                loss = F.mse_loss(noise_pred, noise)  # NB - trying to predict noise (eps) not (noisy_ims-clean_ims) or just (clean_ims)
                pseudo_embeddings = noise_pred

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
    args = parser.parse_args()
    main(args)
