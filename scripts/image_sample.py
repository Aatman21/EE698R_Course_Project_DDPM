# """
# Generate a large batch of image samples from a model and save them as a large
# numpy array. This can be used to produce samples for FID evaluation.
# """

# import argparse
# import os

# import numpy as np
# import torch as th
# import torch.distributed as dist

# from diffusion import dist_util, logger
# from diffusion.script_util import (
#     NUM_CLASSES,
#     model_and_diffusion_defaults,
#     create_model_and_diffusion,
#     add_dict_to_argparser,
#     args_to_dict,
# )


# def main():
#     args = create_argparser().parse_args()

#     dist_util.setup_dist()
#     logger.configure()

#     logger.log("creating model and diffusion...")
#     model, diffusion = create_model_and_diffusion(
#         **args_to_dict(args, model_and_diffusion_defaults().keys())
#     )
#     model.load_state_dict(
#         dist_util.load_state_dict(args.model_path, map_location="cpu")
#     )
#     model.to(dist_util.dev())
#     model.eval()

#     logger.log("sampling...")
#     all_images = []
#     all_labels = []
#     while len(all_images) * args.batch_size < args.num_samples:
#         model_kwargs = {}
#         if args.class_cond:
#             classes = th.randint(
#                 low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
#             )
#             model_kwargs["y"] = classes
#         sample_fn = (
#             diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
#         )
#         sample = sample_fn(
#             model,
#             (args.batch_size, 3, args.image_size, args.image_size),
#             clip_denoised=args.clip_denoised,
#             model_kwargs=model_kwargs,
#         )
#         sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
#         sample = sample.permute(0, 2, 3, 1)
#         sample = sample.contiguous()

#         gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
#         dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
#         all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
#         if args.class_cond:
#             gathered_labels = [
#                 th.zeros_like(classes) for _ in range(dist.get_world_size())
#             ]
#             dist.all_gather(gathered_labels, classes)
#             all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
#         logger.log(f"created {len(all_images) * args.batch_size} samples")

#     arr = np.concatenate(all_images, axis=0)
#     arr = arr[: args.num_samples]
#     if args.class_cond:
#         label_arr = np.concatenate(all_labels, axis=0)
#         label_arr = label_arr[: args.num_samples]
#     if dist.get_rank() == 0:
#         shape_str = "x".join([str(x) for x in arr.shape])
#         out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
#         logger.log(f"saving to {out_path}")
#         if args.class_cond:
#             np.savez(out_path, arr, label_arr)
#         else:
#             np.savez(out_path, arr)

#     dist.barrier()
#     logger.log("sampling complete")


# def create_argparser():
#     defaults = dict(
#         clip_denoised=True,
#         num_samples=10000,
#         batch_size=16,
#         use_ddim=False,
#         model_path="",
#     )
#     defaults.update(model_and_diffusion_defaults())
#     parser = argparse.ArgumentParser()
#     add_dict_to_argparser(parser, defaults)
#     return parser


# if __name__ == "__main__":
#     main()


"""
Run a single noisy image through the diffusion model to clean it and save the result.
"""


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import os
import numpy as np
import torch as th
from PIL import Image
from torchvision import transforms
from diffusion import dist_util, logger
from diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    # Set up logger
    logger.configure()

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    # Load the checkpoint
    model.load_state_dict(th.load(args.model_path, map_location="cpu"), strict=False)
    model.to("cpu")  # Run on CPU
    model.eval()
    

    logger.log("Loading noisy input image...")
    
    # Load the noisy image
    noisy_image = Image.open(args.input_image).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
    ])
    noisy_image_tensor = transform(noisy_image).unsqueeze(0)  # Add batch dimension (1, 3, H, W)

    logger.log("Running the model on the noisy image...")

    # Denoising process
    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )
    denoised_image = sample_fn(
        model,
        noisy_image_tensor.shape,  # Shape of the noisy image tensor
        clip_denoised=args.clip_denoised,
        model_kwargs={"x_start": noisy_image_tensor},  # Pass the noisy image as x_start
    )
    
    # Convert the denoised image to a valid format
    denoised_image = ((denoised_image + 1) * 127.5).clamp(0, 255).to(th.uint8)
    denoised_image = denoised_image.permute(0, 2, 3, 1)  # (1, H, W, 3)
    denoised_image = denoised_image.contiguous().cpu().numpy()

    # Save the denoised image
    out_path = os.path.join(logger.get_dir(), "denoised_image.png")
    logger.log(f"Saving the denoised image to {out_path}")
    Image.fromarray(denoised_image[0]).save(out_path)

    logger.log("Denoising complete.")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        use_ddim=False,  # Default is to use normal p_sample_loop
        model_path=r"C:\Users\Aatman J\Desktop\improvedDDPM\final_code\cifar10_uncond_50M_500K.pt",  # Path to your checkpoint
        input_image=r"C:\Users\Aatman J\Desktop\improvedDDPM\final_code\dog_00446.png",  # Input noisy image path
        image_size=32  # Adjust if needed (32 for CIFAR-10)
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
