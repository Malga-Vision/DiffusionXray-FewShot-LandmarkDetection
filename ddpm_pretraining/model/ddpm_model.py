""" 
Diffusion model architecture modified from the original code  
https://github.com/dome272/Diffusion-Models-pytorch/tree/main 
and
https://huggingface.co/blog/annotated-diffusion
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import logging
from model.nn_blocks import *
from torch.optim import Adam, AdamW, SGD
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics import MeanSquaredError
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import copy

class DDPM:
    def __init__(
        self,
        image_size: int = 256,
        channels: int = 3,
        device = "cuda",
        lr: float = 1e-4,
        optimizer: str = "adam",
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        unet_channels: list = [1, 2, 4, 8],
        unet_self_condition: bool = True,
        unet_att_res: int = 32,
        unet_att_heads: int = 4,
        unet_res_blocks: int = 4,
        use_ema: bool = False,
    ):

        self.timesteps = timesteps
        self.device = device
        self.image_size = image_size
        self.channels = channels
        self.lr = lr

        self.beta_start = beta_start
        self.beta_end = beta_end

        # define beta schedule
        self.beta = self.prepare_noise_schedule(beta_schedule=beta_schedule).to(device)
        
        # define alphas
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        # define model
        self.model = Unet(
            dim=image_size,
            channels=channels,
            dim_mults=unet_channels,
            self_condition=unet_self_condition,
            resnet_block_groups=unet_res_blocks,
            att_heads=unet_att_heads,
            att_res=unet_att_res,
        )
        self.model.to(self.device)

        # define optimizer
        self.optimizer = self.prepare_optimizer(optimizer)
    
        # Initialize EMA
        if use_ema:
            self.ema = EMA(0.995)
            self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False).to(self.device)
        else:
            self.ema = None
            self.ema_model = None

    def update_ema(self):
        # Update the EMA model parameters
        self.ema.update_model_average(self.ema_model, self.model)
        
    def prepare_optimizer(self, optimizer):
        if optimizer == "adam":
            return Adam(self.model.parameters(), lr=self.lr)
        elif optimizer == "adamw":
            return AdamW(self.model.parameters(), lr=self.lr)
        elif optimizer == "sgd":
            return SGD(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError()
        
    def prepare_noise_schedule(self, beta_schedule):
        if beta_schedule == "linear":
            return linear_beta_schedule(
                timesteps=self.timesteps, beta_start=self.beta_start, beta_end=self.beta_end
            )
        elif beta_schedule == "cosine":
            return cosine_beta_schedule(timesteps=self.timesteps)
        elif beta_schedule == "quadratic":
            return quadratic_beta_schedule(
                timesteps=self.timesteps, beta_start=self.beta_start, beta_end=self.beta_end
            )
        elif beta_schedule == "sigmoid":
            return sigmoid_beta_schedule(
                timesteps=self.timesteps, beta_start=self.beta_start, beta_end=self.beta_end
            )
        else:
            raise NotImplementedError


    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    
    def noise_images_conditioned(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.timesteps, size=(n,)).to(self.device)

    def p_losses(self, x, t, loss_type="l1"):
        
        x_t, noise = self.noise_images(x, t)
        predicted_noise = self.model(x_t, t)

        if loss_type == "l1":
            #loss = nn.L1Loss()(noise, predicted_noise)
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == "l2":
            #loss = nn.MSELoss()(noise, predicted_noise)
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            #loss = nn.SmoothL1Loss()(noise, predicted_noise)
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss


    def sample(self, model, batch_size, timesteps=None, x_cond=None):
        with torch.no_grad():
            if timesteps is None:
                timesteps = self.timesteps

            if x_cond is None:
                x = torch.randn((batch_size, self.channels, self.image_size, self.image_size)).to(self.device)
            else:
                x,_ = self.noise_images_conditioned(x_cond, timesteps-1)

            for i in tqdm(reversed(range(1, timesteps)), position=0):
                t = (torch.ones(batch_size) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]

                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        return x
    
    @torch.no_grad()
    def save_noising_process_image(self, x_start, filename):

        images = [x_start[0:1]] # Select the first image from the batch
        timesteps_to_visualize = [i for i in range(0, self.timesteps, self.timesteps // 10)]
        
        for t in timesteps_to_visualize:
            t_tensor = torch.tensor([t], device=self.device)
            noised_image, _ = self.noise_images(x_start, t_tensor)
            images.append(noised_image[0:1]) # Select the first image from the batch

        # Create a grid of images
        image_grid = make_grid(torch.cat(images), nrow=len(timesteps_to_visualize) + 1, normalize=False)
        image_grid = image_grid.clamp(0, 1)

        # Convert to numpy array and transpose axes to HWC format for plotting
        np_image_grid = image_grid.cpu().numpy().transpose((1, 2, 0))

        # Plot and save the image
        plt.figure(figsize=(len(images) * 2, 2))
        plt.imshow(np_image_grid)
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

















