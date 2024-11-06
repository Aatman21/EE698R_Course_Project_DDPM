# CIFAR-10 Diffusion Model Implementation

This repository contains the implementation of the Denoising Diffusion Probabilistic Model (DDPM) applied to the CIFAR-10 dataset. The core objective of this project was to reproduce the results of the "Denoising Diffusion Probabilistic Models" paper by running it on a CPU setup. Significant modifications were made to ensure the code is optimized for a single CPU instead of distributed training, as described below.

## Key Modifications

- **Optimized for single-CPU training**: The original implementation is designed for distributed GPU training. I have adapted the codebase to run efficiently on a single CPU setup by adjusting the batch size, micro-batching, and memory handling processes. This allows the model to be trained on lower-end machines without requiring large-scale hardware setups.
  
- **Results after 20,000 steps**: I trained the model for 20,000 steps and achieved the following metrics, which are saved in the attached `results_lin_sched.xlsx` file also I have trained the model with the polynomial variance which I have attached with the name `results_poly_sched.xlsx`. This includes gradients, loss functions, MSE values, and more, detailed across multiple checkpoints during training.

- **Polynomial Scheduler**: The polynomial variance scheduler introduces a more flexible control mechanism over the noise accumulation during the forward diffusion process. Unlike the linear schedule, which distributes noise evenly over time, the polynomial scheduler can be fine-tuned to either slow down or accelerate the rate of noise addition. 

## Installation

Follow these steps to set up the environment and run the training:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Aatman21/EE698R_Course_Project_DDPM.git

2. **Set Up the Environment**: 
    Install the necessary dependencies by running:
    ```bash
    pip install -e .

3. **Data Preparation**:
    Run the file named cifar10.py under the dataset folder and it will generate the train and test split data used for training and testing.

4. **Running the Original Model**
    To initiate the training process, navigate to the scripts directory and run the image_train.py file with the following command:
    ```bash
    python image_train.py --image_size 32 --num_channels 64 --num_res_blocks 2 --diffusion_steps 1000 --noise_schedule linear --lr 1e-4 --batch_size 8 --microbatch 4
    ```

    - `--image_size 32`: Specifies the resolution of the images in the dataset (CIFAR-10 images are 32x32 pixels).

    - `--num_channels 64`: The number of channels in the first convolutional layer of the model.

    - `--num_res_blocks 2`: The number of residual blocks in each layer of the model.

    - `--diffusion_steps 1000`: The number of steps in the diffusion process.

    - `--noise_schedule linear`: The type of noise schedule to use (linear).

    - `--lr 1e-4`: Learning rate for the optimizer.

    - `--batch_size 8`: Batch size for training.

    - `--microbatch 4`: Smaller micro-batches to manage memory better on single-CPU machines.

5. **Running the Model with a Polynomial Scheduler** 
    To initiate the training process using the polynomial scheduler, navigate to the scripts directory and run the image_train.py file with the following command:
    ``` bash
    python image_train.py --image_size 32 --num_channels 64 --num_res_blocks 2 --diffusion_steps 1000 --noise_schedule polynomial --lr 1e-4 --batch_size 8 --microbatch 4
    ````


## Results
As mentioned earlier, I trained the model for 20,000 steps.

https://drive.google.com/file/d/1aQUMJLSDcSHhdPNt-WDwZd0R-9ZpKhck/view?usp=sharing

I have shared the implementation details of the model in this link: 
https://drive.google.com/file/d/14uiZldN9xbmFjf9_301x3G5XXTrydN9a/view?usp=sharing

## Logging
grad_norm: Gradient norm to monitor gradient updates.
loss: Overall loss of the model.
mse: Mean Squared Error over the generated samples.
These metrics are saved at each checkpoint and logged into the results.xlsx file in this repository. You can access it for detailed insights into how the training evolved over time via this link : 

For `results_lin_sched`
https://drive.google.com/file/d/1G30y9nGM1ri6LyDByNwJiKn6fdXzHwpt/view?usp=sharing

For `results_poly_sched`
https://drive.google.com/file/d/1dC-HwUu8jJ6ElTURb9nJvHUKtXE4hsrI/view?usp=sharing


