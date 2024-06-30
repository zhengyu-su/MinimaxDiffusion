"""
Sample new images from a pre-trained DiT.
"""
import os
import torch
from tqdm import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image, make_grid
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
from diffusers import DDPMScheduler, UNet2DModel, UNet2DConfig
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset == 'imagenet':
    # Labels to condition the model
        with open('./misc/class_indices.txt', 'r') as fp:
            all_classes = fp.readlines()
        all_classes = [class_index.strip() for class_index in all_classes]
        if args.spec == 'woof':
            file_list = './misc/class_woof.txt'
        elif args.spec == 'nette':
            file_list = './misc/class_nette.txt'
        else:
            file_list = './misc/class100.txt'
        with open(file_list, 'r') as fp:
            sel_classes = fp.readlines()

        phase = max(0, args.phase)
        cls_from = args.nclass * phase
        cls_to = args.nclass * (phase + 1)
        sel_classes = sel_classes[cls_from:cls_to]
        sel_classes = [sel_class.strip() for sel_class in sel_classes]
        class_labels = []
        
        for sel_class in sel_classes:
            class_labels.append(all_classes.index(sel_class))

    '''
    # ImagNet classes for cifar10
    if args.dataset == 'cifar10':
        class_labels = [404, 436, 94, 281, 352, 207, 32, 339, 510, 864]
        sel_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    '''
    if args.dataset == 'cifar10':
        class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        sel_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    if args.dataset == 'imagenet':
        latent_size = args.image_size // 8
        model = DiT_models[args.model](
            input_size=latent_size,
            num_classes=args.num_classes
        ).to(device)
        # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
        ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
        print(f"Loading model from {ckpt_path}")
        state_dict = find_model(ckpt_path)
        model.load_state_dict(state_dict, strict=False)
        model.eval()  # important!
        diffusion = create_diffusion(str(args.num_sampling_steps))
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

        batch_size = 1

        for class_label, sel_class in zip(class_labels, sel_classes):
            os.makedirs(os.path.join(args.save_dir, sel_class), exist_ok=True)
            for shift in tqdm(range(args.num_samples // batch_size)):
                # Create sampling noise:
                z = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
                y = torch.tensor([class_label], device=device)

                # Setup classifier-free guidance:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * batch_size, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

                # Sample images:
                samples = diffusion.p_sample_loop(
                    model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
                )
                # Save the samples:
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
                samples = vae.decode(samples / 0.18215).sample

                # Save and display images:
                for image_index, image in enumerate(samples):
                    save_image(image, os.path.join(args.save_dir, sel_class,
                                                f"{image_index + shift * batch_size + args.total_shift}.png"), normalize=True, value_range=(-1, 1))
        
    if args.dataset == 'cifar10':
    	# Load model
        image_size = args.image_size
        model = UNet2DModel.from_pretrained('google/ddpm-cifar10-32', use_safetensors=True)
        model.config.class_embed_type = 'identity'
        model.to('cuda')
        checkpoint = torch.load(args.ckpt)
        print(f"Loading model from {args.ckpt}")
        model.load_state_dict(checkpoint['model'])
        print('model config:', model.config)
        
        # Load scheduler
        scheduler = DDPMScheduler.from_pretrained('google/ddpm-cifar10-32')
        scheduler.set_timesteps(num_inference_steps=1000)
        diffusion = create_diffusion(str(args.num_sampling_steps))
        batch_size = 1
        
        for class_label, sel_class in zip(class_labels, sel_classes):
            os.makedirs(os.path.join(args.save_dir, sel_class), exist_ok=True)
            for shift in tqdm(range(args.num_samples // batch_size)):
                x = torch.randn((batch_size, 3, image_size, image_size), device=device)
                y = torch.tensor([class_label], device=device)
        
                model_input = x

        
                for i, t in tqdm(enumerate(scheduler.timesteps)):
                    with torch.no_grad():
                        noisy_residual = model(model_input, t, y).sample
                    scheduler_output = scheduler.step(noisy_residual, t, model_input)
                    model_input = scheduler_output.prev_sample
            
                # Post-process the image
                image = (model_input/2+0.5).clamp(0,1).squeeze()
                image = (image.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
                image = Image.fromarray(image)
                image.save(os.path.join(args.save_dir, sel_class, f"{shift + args.total_shift}.png"))
        
        '''
        for class_label, sel_class in zip(class_labels, sel_classes):
            os.makedirs(os.path.join(args.save_dir, sel_class), exist_ok=True)
            print('class_label:', class_label, 'sel class:', sel_class)
            for shift in tqdm(range(args.num_samples // batch_size)):
                # Create sampling noise:
                x = torch.randn((batch_size, 3, image_size, image_size), device=device)
                y = torch.tensor([class_label], device=device)
                model_input = x
                # Denoising loop
                for t in scheduler.timesteps:
                    with torch.no_grad():
                    	noisy_residual = model(model_input, t).sample
                    previous_noisy_sample = scheduler.step(noisy_residual, t, model_input).prev_sample
                    model_input = previous_noisy_sample
                    
                image = (x/2+0.5).clamp(0,1).squeeze()
                image = (image.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
                image = Image.fromarray(image)
                # save_image(image, 'ddpm_generated_image.png')
                image.save('ddpm_generated_image.png')
            break

                # Sample images:
                samples = diffusion.p_sample_loop(
                    model, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
                )
                # Save the samples:
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
                #samples = vae.decode(samples / 0.18215).sample

                # Save and display images:
                for image_index, image in enumerate(samples):
                    save_image(image, os.path.join(args.save_dir, sel_class,
                                                f"{image_index + shift * batch_size + args.total_shift}.png"), normalize=True, value_range=(-1, 1))


               
                # Save and display images:
                for image_index, image in enumerate(samples):
                    save_image(image, os.path.join(args.save_dir, sel_class,
                                                f"{image_index + shift * batch_size + args.total_shift}.png"), normalize=True, value_range=(-1, 1))
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--model", type=str, default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[32, 256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--spec", type=str, default='none', help='specific subset for generation')
    parser.add_argument("--save-dir", type=str, default='../logs/test', help='the directory to put the generated images')
    parser.add_argument("--num-samples", type=int, default=100, help='the desired IPC for generation')
    parser.add_argument("--total-shift", type=int, default=0, help='index offset for the file name')
    parser.add_argument("--nclass", type=int, default=10, help='the class number for generation')
    parser.add_argument("--phase", type=int, default=0, help='the phase number for generating large datasets')
    parser.add_argument("--dataset", type=str, default='imagenet', help='the dataset for generation')
    args = parser.parse_args()
    main(args)
