

# ------------------------------------------------------------------------
#                               Libraries
# ------------------------------------------------------------------------

# General libraries
import numpy as np
import logging
from typing import Dict, Any
import time
import os

# Deep learning libraries
#import tensorflow as tf
import torch
import torch.nn as nn

from tqdm import tqdm
from torchmetrics import MeanSquaredError
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance

# Custom libraries
from utils import *
from model.nn_blocks import *

from model.ddpm_model import DDPM



def initialize_ddpm(config: Dict[str, Any], phase: str, device: torch.device) -> DDPM:
    
    if phase == "train":
        ddpm = DDPM(
            image_size=config["dataset"]["image_size"],
            channels=config["dataset"]["image_channels"],
            device=device,
            lr=config["model"]["lr"],
            optimizer=config["model"]["optimizer"],
            timesteps=config["model"]["beta_schedule"]["train"]["n_timestep"],
            beta_schedule=config["model"]["beta_schedule"]["train"]["schedule"],
            beta_start=config["model"]["beta_schedule"]["train"]["linear_start"],
            beta_end=config["model"]["beta_schedule"]["train"]["linear_end"],
            unet_self_condition=config["model"]["unet"]["self_condition"],
            unet_channels=config["model"]["unet"]["channel_mults"],
            unet_res_blocks=config["model"]["unet"]["res_blocks"],
            unet_att_heads=config["model"]["unet"]["num_head_channels"],
            unet_att_res=config["model"]["unet"]["attn_res"],
            use_ema=config["model"]["use_ema"],
        )
        
    elif phase == "test":
        ddpm = DDPM(
            image_size=config["dataset"]["image_size"],
            channels=config["dataset"]["image_channels"],
            device=device,
            lr=config["model"]["lr"],
            optimizer=config["model"]["optimizer"],
            timesteps=config["model"]["beta_schedule"]["test"]["n_timestep"],
            beta_schedule=config["model"]["beta_schedule"]["test"]["schedule"],
            beta_start=config["model"]["beta_schedule"]["test"]["linear_start"],
            beta_end=config["model"]["beta_schedule"]["test"]["linear_end"],
            unet_self_condition=config["model"]["unet"]["self_condition"],
            unet_channels=config["model"]["unet"]["channel_mults"],
            unet_res_blocks=config["model"]["unet"]["res_blocks"],
            unet_att_heads=config["model"]["unet"]["num_head_channels"],
            unet_att_res=config["model"]["unet"]["attn_res"],
        )
    else:
        raise ValueError(f"Phase {phase} is not valid. Must be either 'train' or 'test'")

    return ddpm

def save_model(model, optimizer, epoch, n_iter, loss, save_path, name="last_model"):
    torch.save({
        "epoch": epoch,
        "n_iter": n_iter,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, f"{save_path}/{name}.pt")

def save_ema_model(ema_model, epoch, n_iter, save_path, name="last_ema_model"):
    torch.save({
        "epoch": epoch,
        "n_iter": n_iter,
        "model_state_dict": ema_model.state_dict(),
    }, f"{save_path}/{name}.pt")


def load_model(save_model_path, device, ddpm):
    assert os.path.exists(f"{save_model_path}"), f"Model {save_model_path} does not exist"
    checkpoint = torch.load(f"{save_model_path}", map_location=device)
    ddpm.model.load_state_dict(checkpoint["model_state_dict"])
    ddpm.optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) if "optimizer_state_dict" in checkpoint else None
    epoch = checkpoint.get('epoch', 'undefined')
    n_iter = checkpoint.get('n_iter', 'undefined')
    loss = checkpoint.get('loss', np.inf)
    del checkpoint
    print(f"Model loaded for epoch: {epoch} and iteration: {n_iter} with loss: {loss}")
    return epoch, n_iter, loss

def load_ema_model(save_model_path, device, ddpm):
    assert os.path.exists(f"{save_model_path}"), f"Model {save_model_path} does not exist"
    ema_checkpoint = torch.load(f"{save_model_path}", map_location=device)
    ddpm.ema_model.load_state_dict(ema_checkpoint["model_state_dict"])
    ema_epoch = ema_checkpoint.get('epoch', 0)
    ema_n_iter = ema_checkpoint.get('n_iter', 0)
    del ema_checkpoint
    print(f"EMA Model loaded for epoch: {ema_epoch} and iteration: {ema_n_iter}")
    return ema_epoch, ema_n_iter


# ------------------------------------------------------------------------
#                               TRAINING DIFFUSION MODEL
# ------------------------------------------------------------------------


def train_diffusion_model(config, train_dataloader, save_model_path, root_path, device, continue_training=False):
    # config params
    image_size = config["dataset"]["image_size"]
    channels = config["dataset"]["image_channels"]
    batch_size = config["dataset"]["batch_size"]
    grad_accumulation = config["dataset"]["grad_accumulation"]
    iterations = config["model"]["iterations"]
    # Compute number of epochs based on the number of iterations and the number of batches and gradient accumulation
    # Total epochs = iterations / iterations per epoch (batches per epoch / grad_accumulation)
    epochs = int(iterations / (len(train_dataloader) / grad_accumulation)) + 1
    
    loss_type = config["model"]["loss_type"]
    timesteps=config["model"]["beta_schedule"]["train"]["n_timestep"]
    freq_metrics = config["model"]["freq_metrics"]
    freq_checkpoint = config["model"]["freq_checkpoint"]
    use_ema = config["model"]["use_ema"]

    # instanciate the diffusion model
    ddpm = initialize_ddpm(config, phase="train", device=device)

    # Print model count trainable parameters
    table, total_params = count_parameters(ddpm.model)
    logging.info(f"Total Trainable Params: {total_params}")
    
    # If continue_training is True, load the model
    if continue_training:
        model_path = os.path.join(save_model_path, "last_model.pt")
        start_epoch, n_iter, best_loss = load_model(model_path, device, ddpm)
        
        if use_ema:
            ema_model_path = os.path.join(save_model_path, "last_ema_model.pt")
            ema_epoch, ema_n_iter = load_ema_model(ema_model_path, device, ddpm)
        
    else:
        start_epoch, n_iter = 0, 0
        best_loss = np.inf
        
    # instanciate metrics
    ssim = StructuralSimilarityIndexMeasure(data_range=None, reduction="elementwise_mean")
    mse = MeanSquaredError()
    fid = FrechetInceptionDistance(normalize=True)
    
    # Start training
    start_time = time.time()

    try:    
        # Train diffusion model
        for epoch in tqdm(range(start_epoch, epochs), initial=start_epoch, total=epochs, desc="Epoch"):
            epoch_loss = 0.0
            torch.cuda.empty_cache()
            
            for batch_idx, data in enumerate(tqdm(train_dataloader, desc="Batch", leave=False)):

                # Load data
                x = data['image'].to(device)
                x_names = data['name']
                batch_size = x.shape[0]
                
                # Save noising process image
                if batch_idx == 0 and epoch == 0:
                    ddpm.save_noising_process_image(x, f'{root_path}/noising_process.png')

                # Forward pass
                t = ddpm.sample_timesteps(batch_size)
                loss = ddpm.p_losses(x=x, t=t, loss_type=loss_type)

                # Backward pass 
                loss.backward()
                loss = loss / grad_accumulation  # Normalize loss to account for batch accumulation
                epoch_loss += loss.item()
                
                # Update weights
                if ((batch_idx + 1) % grad_accumulation == 0) or (batch_idx + 1 == len(train_dataloader)):
                
                    # Clip to solve the gradient exploding problem
                    nn.utils.clip_grad_value_(ddpm.model.parameters(), clip_value=1.0)

                    ddpm.optimizer.step()
                    ddpm.optimizer.zero_grad()
                
                    # Update EMA parameters after each training step
                    if use_ema: ddpm.update_ema()
                    
                    # Update number of steps                
                    n_iter += 1    

                    # Save model checkpoint after each training step
                    save_model(ddpm.model, ddpm.optimizer, epoch, n_iter, loss, save_model_path, name="last_model")
                    if use_ema: save_ema_model(ddpm.ema_model, epoch, n_iter, save_model_path, name="last_ema_model")
        
                # Compute Metrics
                if n_iter % freq_metrics == 0 and batch_idx % grad_accumulation == 0 and n_iter != 0:
                    # Set model to eval mode
                    ddpm.model.eval()
                                            
                    # Generate images conditioned on the input images x
                    x_hat = ddpm.sample(model=ddpm.model, x_cond=x, batch_size=batch_size, timesteps=timesteps)
                    if use_ema: ema_x_hat = ddpm.sample(model=ddpm.ema_model, x_cond=x, batch_size=batch_size, timesteps=timesteps)
                
                    # Detach tensors from GPU
                    x = x.detach().cpu()
                    x_hat = x_hat.detach().cpu()
                    
                    # Check pixel range of generated images
                    x_hat_min, x_hat_max = check_pixels_range_of_image(x_hat)
                    
                    # Compute metrics
                    ssim_metric = ssim(x_hat, x)
                    mse_metric = mse(x_hat, x)
                    
                    # Compute FID
                    real_images = x if channels == 3 else x.repeat(1, 3, 1, 1)
                    fake_images = x_hat if channels == 3 else x_hat.repeat(1, 3, 1, 1)

                    fid.update(real_images, real=True)
                    fid.update(fake_images.clamp(0, 1), real=False)
                    fid_score = fid.compute()
                    
                    # Compute differece images
                    diff = compute_diff(real_images, fake_images)
                    
                    # Save batch original images in the first row, generated images in the second row, and diff images in the third row
                    train_imgs_path = generate_path(f"{root_path}/images/train")
                    image_titles = [f"{x_names[i]}" for i in range(batch_size)]
                    save_images([real_images, fake_images, diff], f"{train_imgs_path}/train_epoch{epoch}_iteration{n_iter}_batch{batch_idx}.jpg", f"Epoch {epoch} - Iteration {n_iter} - Batch {batch_idx} - Timesteps {timesteps}", image_titles)

                    # Log metrics
                    logging.info(f"\nEpoch/Iteration {epoch}/{n_iter} \t Batch {batch_idx} \t Loss: {loss.item():.6f}")
                    logging.info(f"\t\t SSIM: {ssim_metric.item():.4f} \t MSE: {mse_metric.item():.6f} \t FID: {fid_score:.2f} \t Pixel range: [{x_hat_min:.2f}, {x_hat_max:.2f}]")

                    
                    # Send message
                    message = (
                        f"<b>Epoch/Iteration {epoch}/{n_iter}</b> --> [{x_hat_min:.2f}, {x_hat_max:.2f}] \n"
                        f"  • <b>Loss:</b> {loss.item():.4f} \n"
                        f"  • <b>SSIM:</b> {ssim_metric.item():.4f} \n"
                        f"  • <b>MSE:</b> {mse_metric.item():.4f} \n"
                        f"  • <b>FID:</b> {fid_score:.4f}"
                    )
                            
                    # Compute EMA metrics
                    if use_ema:

                        ema_x_hat = ema_x_hat.detach().cpu()
                        
                        ema_ssim_metric = ssim(ema_x_hat, x)
                        ema_mse_metric = mse(ema_x_hat, x)

                        ema_x_hat_min, ema_x_hat_max = check_pixels_range_of_image(ema_x_hat)
                        
                        ema_fake_images = ema_x_hat if channels == 3 else ema_x_hat.repeat(1, 3, 1, 1)
                        
                        fid.update(ema_fake_images.clamp(0, 1), real=False)
                        ema_fid_score = fid.compute()
                                
                        ema_diff = compute_diff(real_images, ema_fake_images)


                        # Set image titles for each image 
                        ema_image_titles = [f"{x_names[i]}" for i in range(batch_size)]
                        save_images([real_images, ema_fake_images, ema_diff], f"{train_imgs_path}/train_epoch{epoch}_iteration{n_iter}_batch{batch_idx}_ema.jpg",  f"Epoch {epoch} - Iteration {n_iter} - Batch {batch_idx} - Timesteps {timesteps}", ema_image_titles)
        
                        logging.info(f"\t\t SSIM: {ema_ssim_metric.item():.4f} \t MSE: {ema_mse_metric.item():.6f} \t FID: {ema_fid_score:.2f}\t Pixel range: [{ema_x_hat_min:.2f}, {ema_x_hat_max:.2f}]")

                        
                        message += (
                            f"\n\n<b>EMA Epoch/Iteration {epoch}/{n_iter}</b> --> [{ema_x_hat_min:.2f}, {ema_x_hat_max:.2f}] \n"
                            f"  • <b>Loss:</b> {loss.item():.4f} \n"
                            f"  • <b>SSIM:</b> {ema_ssim_metric.item():.4f} \n"
                            f"  • <b>MSE:</b> {ema_mse_metric.item():.4f} \n"
                            f"  • <b>FID:</b> {ema_fid_score:.4f}"
                        )
                                        
                    
                    # Delete tensors from GPU
                    del x, x_hat, real_images, fake_images, diff, ema_x_hat, ema_fake_images, ema_diff
                    
                    # Log and print message
                    print(message)
                    logging.info(message)
                                        
                    # Set model back to train mode
                    ddpm.model.train()
                                    
                # Save model checkpoint
                if n_iter % freq_checkpoint == 0 and n_iter!= 0 and batch_idx % grad_accumulation == 0:
                    print(f"Saving model checkpoint at epoch {epoch} and iteration {n_iter} with loss: {loss.item():.4f}")                
                    save_model(ddpm.model, ddpm.optimizer, epoch, n_iter, loss, save_model_path, name=f"model_epoch{epoch}_step{n_iter}")
                    if use_ema: save_ema_model(ddpm.ema_model, epoch, n_iter, save_model_path, name=f"ema_model_epoch{epoch}_step{n_iter}")

                    if n_iter % iterations == 0:
                        print(f"Reaching {iterations} iterations. Exiting training...")
                        exit()
                        
            epoch_loss /= len(train_dataloader)
                            
    except (KeyboardInterrupt, SystemExit, Exception) as e:
        if isinstance(e, Exception):
            print(f"Exception: {e}")
        print("\nTraining interrupted. Saving final state...")
        save_model(ddpm.model, ddpm.optimizer, epoch, n_iter, loss, save_model_path, name="last_model")
        if use_ema: save_ema_model(ddpm.ema_model, epoch, n_iter, save_model_path, name="last_ema_model")
        
    finally:
        print(f"Training completed in {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
        logging.info(f"Training completed in {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
        del ddpm
        torch.cuda.empty_cache()