
# ------------------------------------------------------------------------
#                               Libraries 
# ------------------------------------------------------------------------

# General libraries
import os
import argparse
import json
import sys
import time
import logging

import numpy as np

# Deep learning libraries
import torch
#import tensorflow as tf
#import tensorboard as tb
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Custom libraries
from utils import *
from model.training_functions import *

# Set random seed
np.random.seed(42)
torch.manual_seed(42) 
torch.cuda.manual_seed(42)

# ------------------------------------------------------------------------
#                               MAIN
# ------------------------------------------------------------------------

if __name__ == "__main__":
    # Parse arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./ddpm_pretraining/config/config.json",
        help="Path to the JSON config file."
    )

    args = parser.parse_args()
    config = json.load(open(args.config))

    # Print system info
    print("----------------------------------------- SYSTEM INFO -----------------------------------------") 
    print("Python version: {}".format(sys.version))
    print("Pytorch version: {}".format(torch.__version__))
    
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        GPU = os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        GPU = config["gpu"]
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU}"
        
    device = f"cuda" if torch.cuda.is_available() else "cpu"
    print(f"Torch GPU Name: {torch.cuda.get_device_name(0)}... Using GPU {GPU}" if device == "cuda" else "Torch GPU not available... Using CPU")
        
    print("------------------------------------------------------------------------------------------------")

    # Config params for training and testing the model
    root_path = config["experiment_path"]
    DATASET_NAME = config["dataset"]["name"]
    DATASET_PATH = os.path.join(config["dataset"]["path"], DATASET_NAME)
    batch_size = config["dataset"]["batch_size"]
    image_size = config["dataset"]["image_size"]
    image_channels = config["dataset"]["image_channels"]
    pin_memory = config["dataset"]["pin_memory"]
    num_workers = os.cpu_count() if config["dataset"]["num_workers"] == None else config["dataset"]["num_workers"]

    # Create train and test dataloaders
    train_dataloader, test_dataloader = load_data(DATASET_PATH, image_size, image_channels, batch_size, pin_memory=pin_memory, num_workers=num_workers)

    # Save model path and tensorboard writer and path for the experiment
    PREFIX_PATH = f"{root_path}/{DATASET_NAME}/{config['model']['beta_schedule']['train']['schedule']}_{config['model']['beta_schedule']['train']['n_timestep']}/size{image_size}_ch{image_channels}"
    
    # Create log file for the experiment
    if not os.path.exists(f'{PREFIX_PATH}/log_file.txt'):
        os.makedirs(PREFIX_PATH, exist_ok=True)

        with open(f'{PREFIX_PATH}/log_file.txt', 'w') as f:
            pass     
               
    logging.basicConfig(format="%(message)s", level=logging.INFO, filename=f'{PREFIX_PATH}/log_file.txt', filemode='a') # %(asctime)s 

    # Save the original model checkpoint and the ema model checkpoint
    save_model_path = generate_path(f"{PREFIX_PATH}/models/")
    writer = SummaryWriter(f"{PREFIX_PATH}/runs")

    # Print config params
    print("----------------------------------------- CONFIG PARAMS -----------------------------------------")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Dataset size: {len(train_dataloader.dataset)}")
    print(f"Batch size: {batch_size} | Accumulation steps: {config['dataset']['grad_accumulation']}")
    print(f"Number of batches: {len(train_dataloader)}")
    print(f"Image shape: ({image_size}, {image_size}, {image_channels})")
    print(f"Save model path: {save_model_path}")
    print("------------------------------------------------------------------------------------------------")
    
    # Train diffusion model
    print("----------------------------------------- START TRAINING -----------------------------------------")
    print(f"Total epochs: {int(config['model']['iterations'] / (len(train_dataloader) / config['dataset']['grad_accumulation']))} | Total iterations: {config['model']['iterations']} | Iterations per epoch: {len(train_dataloader)/config['dataset']['grad_accumulation']}")
    train_diffusion_model(config, train_dataloader, save_model_path, PREFIX_PATH, writer, device, continue_training=config["model"]["continue_training"])

    print("----------------------------------------- END TRAINING -----------------------------------------")
    