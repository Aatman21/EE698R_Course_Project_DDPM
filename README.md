<!-- # improved-diffusion

This is the codebase for [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672).

# Usage

This section of the README walks through how to train and sample from a model.

## Installation

Clone this repository and navigate to it in your terminal. Then run:

```
pip install -e .
```

This should install the `improved_diffusion` python package that the scripts depend on.

## Preparing Data

The training code reads images from a directory of image files. In the [datasets](datasets) folder, we have provided instructions/scripts for preparing these directories for ImageNet, LSUN bedrooms, and CIFAR-10.

For creating your own dataset, simply dump all of your images into a directory with ".jpg", ".jpeg", or ".png" extensions. If you wish to train a class-conditional model, name the files like "mylabel1_XXX.jpg", "mylabel2_YYY.jpg", etc., so that the data loader knows that "mylabel1" and "mylabel2" are the labels. Subdirectories will automatically be enumerated as well, so the images can be organized into a recursive structure (although the directory names will be ignored, and the underscore prefixes are used as names).

The images will automatically be scaled and center-cropped by the data-loading pipeline. Simply pass `--data_dir path/to/images` to the training script, and it will take care of the rest.

## Training

To train your model, you should first decide some hyperparameters. We will split up our hyperparameters into three groups: model architecture, diffusion process, and training flags. Here are some reasonable defaults for a baseline:

```
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"
```

Here are some changes we experiment with, and how to set them in the flags:

 * **Learned sigmas:** add `--learn_sigma True` to `MODEL_FLAGS`
 * **Cosine schedule:** change `--noise_schedule linear` to `--noise_schedule cosine`
 * **Importance-sampled VLB:** add `--use_kl True` to `DIFFUSION_FLAGS` and add `--schedule_sampler loss-second-moment` to  `TRAIN_FLAGS`.
 * **Class-conditional:** add `--class_cond True` to `MODEL_FLAGS`.

Once you have setup your hyper-parameters, you can run an experiment like so:

```
python scripts/image_train.py --data_dir path/to/images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

You may also want to train in a distributed manner. In this case, run the same command with `mpiexec`:

```
mpiexec -n $NUM_GPUS python scripts/image_train.py --data_dir path/to/images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

When training in a distributed manner, you must manually divide the `--batch_size` argument by the number of ranks. In lieu of distributed training, you may use `--microbatch 16` (or `--microbatch 1` in extreme memory-limited cases) to reduce memory usage.

The logs and saved models will be written to a logging directory determined by the `OPENAI_LOGDIR` environment variable. If it is not set, then a temporary directory will be created in `/tmp`.

## Sampling

The above training script saves checkpoints to `.pt` files in the logging directory. These checkpoints will have names like `ema_0.9999_200000.pt` and `model200000.pt`. You will likely want to sample from the EMA models, since those produce much better samples.

Once you have a path to your model, you can generate a large batch of samples like so:

```
python scripts/image_sample.py --model_path /path/to/model.pt $MODEL_FLAGS $DIFFUSION_FLAGS
```

Again, this will save results to a logging directory. Samples are saved as a large `npz` file, where `arr_0` in the file is a large batch of samples.

Just like for training, you can run `image_sample.py` through MPI to use multiple GPUs and machines.

You can change the number of sampling steps using the `--timestep_respacing` argument. For example, `--timestep_respacing 250` uses 250 steps to sample. Passing `--timestep_respacing ddim250` is similar, but uses the uniform stride from the [DDIM paper](https://arxiv.org/abs/2010.02502) rather than our stride.

To sample using [DDIM](https://arxiv.org/abs/2010.02502), pass `--use_ddim True`.

## Models and Hyperparameters

This section includes model checkpoints and run flags for the main models in the paper.

Note that the batch sizes are specified for single-GPU training, even though most of these runs will not naturally fit on a single GPU. To address this, either set `--microbatch` to a small value (e.g. 4) to train on one GPU, or run with MPI and divide `--batch_size` by the number of GPUs.

Unconditional ImageNet-64 with our `L_hybrid` objective and cosine noise schedule [[checkpoint](https://openaipublic.blob.core.windows.net/diffusion/march-2021/imagenet64_uncond_100M_1500K.pt)]:

```bash
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"
```

Unconditional CIFAR-10 with our `L_hybrid` objective and cosine noise schedule [[checkpoint](https://openaipublic.blob.core.windows.net/diffusion/march-2021/cifar10_uncond_50M_500K.pt)]:

```bash
MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"
```

Class-conditional ImageNet-64 model (270M parameters, trained for 250K iterations) [[checkpoint](https://openaipublic.blob.core.windows.net/diffusion/march-2021/imagenet64_cond_270M_250K.pt)]:

```bash
MODEL_FLAGS="--image_size 64 --num_channels 192 --num_res_blocks 3 --learn_sigma True --class_cond True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 3e-4 --batch_size 2048"
```

Upsampling 256x256 model (280M parameters, trained for 500K iterations) [[checkpoint](https://openaipublic.blob.core.windows.net/diffusion/march-2021/upsample_cond_500K.pt)]:

```bash
MODEL_FLAGS="--num_channels 192 --num_res_blocks 2 --learn_sigma True --class_cond True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 3e-4 --batch_size 256"
```

LSUN bedroom model (lr=1e-4) [[checkpoint](https://openaipublic.blob.core.windows.net/diffusion/march-2021/lsun_uncond_100M_1200K_bs128.pt)]:

```bash
MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"
```

LSUN bedroom model (lr=2e-5) [[checkpoint](https://openaipublic.blob.core.windows.net/diffusion/march-2021/lsun_uncond_100M_2400K_bs64.pt)]:

```bash
MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --use_scale_shift_norm False"
TRAIN_FLAGS="--lr 2e-5 --batch_size 128"
```

Unconditional ImageNet-64 with the `L_vlb` objective and cosine noise schedule [[checkpoint](https://openaipublic.blob.core.windows.net/diffusion/march-2021/imagenet64_uncond_vlb_100M_1500K.pt)]:

```bash
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --use_kl True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --schedule_sampler loss-second-moment"
```

Unconditional CIFAR-10 with the `L_vlb` objective and cosine noise schedule [[checkpoint](https://openaipublic.blob.core.windows.net/diffusion/march-2021/cifar10_uncond_vlb_50M_500K.pt)]:

```bash
MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule cosine --use_kl True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128 --schedule_sampler loss-second-moment"
``` -->



# CIFAR-10 Diffusion Model Implementation

This repository contains the implementation of the Denoising Diffusion Probabilistic Model (DDPM) applied to the CIFAR-10 dataset. The core objective of this project was to reproduce the results of the "Denoising Diffusion Probabilistic Models" paper by running it on a CPU setup. Significant modifications were made to ensure the code is optimized for a single CPU instead of distributed training, as described below.

## Key Modifications

- **Optimized for single-CPU training**: The original implementation is designed for distributed GPU training. I have adapted the codebase to run efficiently on a single CPU setup by adjusting the batch size, micro-batching, and memory handling processes. This allows the model to be trained on lower-end machines without requiring large-scale hardware setups.
  
- **Results after 30,000 steps**: I trained the model for 30,000 steps and achieved the following metrics, which are saved in the attached `results.xlsx` file. This includes gradients, loss functions, MSE values, and more, detailed across multiple checkpoints during training. You can explore all these results in the Excel sheet linked in this repo.

## Installation

Follow these steps to set up the environment and run the training:

1. **Clone the Repository**:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>

2. **Set Up the Environment**: 
    Install the necessary dependencies by running:
    ```bash
    pip install -e .

3. **Data Preparation**:
    Download the dataset from this link : 

4. **Running the Model**
    To initiate the training process, navigate to the scripts directory and run the image_train.py file with the following command:
    ```bash
    python image_train.py --image_size 32 --num_channels 64 --num_res_blocks 2 --diffusion_steps 1000 --noise_schedule linear --lr 1e-4 --batch_size 8 --microbatch 4
    Explanation of the Key Flags:
    --image_size 32: Specifies the resolution of the images in the dataset (CIFAR-10 images are 32x32 pixels).
    --num_channels 64: The number of channels in the first convolutional layer of the model.
    --num_res_blocks 2: The number of residual blocks in each layer of the model.
    --diffusion_steps 1000: The number of steps in the diffusion process.
    --noise_schedule linear: The type of noise schedule to use (linear).
    --lr 1e-4: Learning rate for the optimizer.
    --batch_size 8: Batch size for training.
    --microbatch 4: Smaller micro-batches to manage memory better on single-CPU machines.

5. **Results and Logging**
    As mentioned earlier, I trained the model for 30,000 steps. The training loop outputs a number of metrics, including:

    grad_norm: Gradient norm to monitor gradient updates.
    loss: Overall loss of the model.
    mse: Mean Squared Error over the generated samples.
    These metrics are saved at each checkpoint and logged into the results.xlsx file in this repository. You can access it for detailed insights into how the training evolved over time via this link : 


