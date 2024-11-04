import torch
import numpy as np
import torchvision
from builders.moco import MoCo
from builders.mocov2 import MoCoV2
from builders.mocov3 import MoCoV3
from builders.simclr import SimCLR
from builders.simclrv2 import SimCLRv2
from builders.dino import DINO
from builders.byol import BYOL
from builders.barlow_twins import BarlowTwins
from segmentation_models_pytorch import Unet as smpUnet
from utils import *
from ssl_datasets import *
import logging
import os
import sys
from tqdm import tqdm
import time
import argparse
import json
import signal
import sys
import tempfile

# Set random seed
np.random.seed(42)
torch.manual_seed(42) 
torch.cuda.manual_seed(42)



def safe_save(state, filename):
    """
    Safely save a model state to a file using atomic operations.
    
    Args:
    state (dict): The state to be saved (usually containing model and optimizer states)
    filename (str): The name of the file to save the state to
    """
    # Create a temporary file
    temp_filename = None
    try:
        # Create a temporary file in the same directory as the target file
        directory = os.path.dirname(filename)
        with tempfile.NamedTemporaryFile(delete=False, dir=directory) as tmp_file:
            temp_filename = tmp_file.name
            # Save the state to the temporary file
            torch.save(state, temp_filename)
        
        # If the save was successful, rename the temporary file to the target filename
        # This operation is atomic on most Unix-like systems
        os.replace(temp_filename, filename)
        #print(f"Model safely saved to {filename}")
    except Exception as e:
        print(f"Error during safe save: {e}")
        # If there was an error, remove the temporary file if it was created
        if temp_filename and os.path.exists(temp_filename):
            os.remove(temp_filename)

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
        default="config/config.json",
        help="Path to the JSON config file."
    )

    args = parser.parse_args()
    config = json.load(open(args.config))

    # Config params for training and testing the model
    ROOT_PATH = config["experiment_path"]
    DATASET_NAME = config["dataset"]["name"]
    DATASET_PATH = os.path.join(config["dataset"]["path"], DATASET_NAME)
    IMAGE_SIZE = config["dataset"]["image_size"]
    IMAGE_CHANNELS = config["dataset"]["image_channels"]
    BATCH_SIZE = config["dataset"]["batch_size"]
    GRAD_ACCUMULATION = config["dataset"]["grad_accumulation"]
    PIN_MEMORY = config["dataset"]["pin_memory"]
    NUM_WORKERS = os.cpu_count() if config["dataset"]["num_workers"] == None else config["dataset"]["num_workers"]
    
    # SSL training params
    LR = config["model"]["lr"]
    EPOCHS = config["model"]["epochs"]
    SSL_METHOD = config["model"]["name"]
    BACKBONE_NAME = config["model"]["encoder"]
    OPTIMIZER = config["model"]["optimizer"]
    
    # Load the dataset
    train_dataloader, test_dataloader = load_data(DATASET_PATH, IMAGE_SIZE, IMAGE_CHANNELS, BATCH_SIZE, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)

    # Save model path and tensorboard writer and path for the experiment
    PREFIX_PATH = f"{ROOT_PATH}/{DATASET_NAME}/size{IMAGE_SIZE}_ch{IMAGE_CHANNELS}"

    # Create log file for the experiment
    if not os.path.exists(f'{PREFIX_PATH}/models/{SSL_METHOD}/{BACKBONE_NAME}/log_file.txt'):
        os.makedirs(f"{PREFIX_PATH}/models/{SSL_METHOD}/{BACKBONE_NAME}/", exist_ok=True)

        with open(f'{PREFIX_PATH}/models/{SSL_METHOD}/{BACKBONE_NAME}/log_file.txt', 'w') as f:
            pass     

    save_model_path = generate_path(f"{PREFIX_PATH}/models/{SSL_METHOD}/{BACKBONE_NAME}")
    logging.basicConfig(format="%(message)s", level=logging.INFO, filename=f'{PREFIX_PATH}/models/{SSL_METHOD}/{BACKBONE_NAME}/log_file.txt', filemode='a') # %(asctime)s 


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

    print("----------------------------------------- CONFIG INFO -----------------------------------------")
    print(f"Dataset Name: {DATASET_NAME}")
    print(f"Dataset Path: {DATASET_PATH}")
    print(f"Image Size: ({IMAGE_SIZE}, {IMAGE_SIZE}, {IMAGE_CHANNELS})")
    print(f"Batch Size: {BATCH_SIZE} with cumulation of {GRAD_ACCUMULATION}")
    print(f"SSL Method: {SSL_METHOD}")
    print(f"Epochs: {EPOCHS}")
    print(f"Starting Learning Rate: {LR}")
    print(f"Save Model Path: {save_model_path}")
    print("------------------------------------------------------------------------------------------------")

    print("----------------------------------------- START TRAINING -----------------------------------------")

    # Initialize backbones for self-supervised learning method
    if BACKBONE_NAME == "resnet18":
        backbone = torchvision.models.resnet18(weights=None)
        # Change the last layer of the backbone to Identity       
        feature_size = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()
    
    elif BACKBONE_NAME == "resnet34":
        backbone = torchvision.models.resnet34(weights=None)
        feature_size = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()
        
    elif BACKBONE_NAME == "resnet50":
        backbone = torchvision.models.resnet50(weights=None)
        feature_size = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()
        
    elif BACKBONE_NAME == "resnet101":
        backbone = torchvision.models.resnet101(weights=None)
        feature_size = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()
        
    elif BACKBONE_NAME == "resnet152":
        backbone = torchvision.models.resnet152(weights=None)
        feature_size = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()
        
    elif BACKBONE_NAME == "resnext50_32x4d":
        backbone = torchvision.models.resnext50_32x4d(weights=None)
        feature_size = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()
        
    elif BACKBONE_NAME == "resnext101_32x8d":
        backbone = torchvision.models.resnext101_32x8d(weights=None)
        feature_size = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()
        
    elif BACKBONE_NAME == "vgg11":
        backbone = torchvision.models.vgg11(weights=None)
        feature_size = backbone.classifier[6].in_features
        backbone.classifier[6] = torch.nn.Identity()
        
    elif BACKBONE_NAME == "vgg13":
        backbone = torchvision.models.vgg13(weights=None)
        feature_size = backbone.classifier[6].in_features
        backbone.classifier[6] = torch.nn.Identity()
        
    elif BACKBONE_NAME == "vgg16":
        backbone = torchvision.models.vgg16(weights=None)
        feature_size = backbone.classifier[6].in_features
        backbone.classifier[6] = torch.nn.Identity()
        
    elif BACKBONE_NAME == "vgg19":
        backbone = torchvision.models.vgg19(weights=None)
        feature_size = backbone.classifier[6].in_features
        backbone.classifier[6] = torch.nn.Identity()
        
    elif BACKBONE_NAME == "densenet121":
        backbone = torchvision.models.densenet121(weights=None)
        feature_size = backbone.classifier.in_features
        backbone.classifier = torch.nn.Identity()
        
    elif BACKBONE_NAME == "densenet169":
        backbone = torchvision.models.densenet169(weights=None)
        feature_size = backbone.classifier.in_features
        backbone.classifier = torch.nn.Identity()
        
    elif BACKBONE_NAME == "densenet201":
        backbone = torchvision.models.densenet201(weights=None)
        feature_size = backbone.classifier.in_features
        backbone.classifier = torch.nn.Identity()
        
    elif BACKBONE_NAME == "densenet161":
        backbone = torchvision.models.densenet161(weights=None)
        feature_size = backbone.classifier.in_features
        backbone.classifier = torch.nn.Identity()
    else:
        raise ValueError(f"Unknown backbone: {BACKBONE_NAME}")


    # Initialize ssl method model and optimizer
    if SSL_METHOD == "moco":
        model = MoCo(backbone, feature_size, projection_dim=128, K=65536, m=0.999, temperature=0.07, image_size=IMAGE_SIZE)
    elif SSL_METHOD == "mocov2":
        model = MoCoV2(backbone, feature_size, projection_dim=128, K=65536, m=0.999, temperature=0.07, image_size=IMAGE_SIZE)
    elif SSL_METHOD == "mocov3":
        model = MoCoV3(backbone, feature_size, projection_dim=256, hidden_dim=2048, temperature=0.5, m=0.999, image_size=IMAGE_SIZE)  
              
    elif SSL_METHOD == "simclr":
        model = SimCLR(backbone, feature_size, projection_dim=128, temperature=0.5, image_size=IMAGE_SIZE)
    elif SSL_METHOD == "simclrv2":
        model = SimCLRv2(backbone, feature_size, projection_dim=128, temperature=0.5, image_size=IMAGE_SIZE)
        
    elif SSL_METHOD == "dino":
        model = DINO(backbone, feature_size, projection_dim=256, hidden_dim=2048, bottleneck_dim=256, temp_s=0.1, temp_t=0.5, m=0.5, lamda=0.996, num_crops=6) # image_size=IMAGE_SIZE)
    
    elif SSL_METHOD == "byol":
        model = BYOL(backbone, feature_size, projection_dim=256, hidden_dim=4096, tau=0.996, image_size=IMAGE_SIZE)
        
    elif SSL_METHOD == "barlow_twins":
        model = BarlowTwins(backbone, feature_size, projection_dim=8192, hidden_dim=8192, lamda=0.005, image_size=IMAGE_SIZE)
        
    else:
        raise ValueError(f"Unknown SSL method: {SSL_METHOD}")


    model = model.to(device)
    
    if OPTIMIZER == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    elif OPTIMIZER == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER} - Choose either 'adam' or 'adamw'")
    
    
    # Check if there are any saved models in the save_model_path
    if os.path.exists(f'{save_model_path}/last_model.pth'):
        
        # Get the latest model weights
        model_path = f'{save_model_path}/last_model.pth'
        checkpoint = torch.load(model_path, map_location=device)    

        # Load the model weights and optimizer state
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']
        n_iter = checkpoint['iteration']
        print(f"Loading model weights from epoch {starting_epoch} and iteration {n_iter}") 
        del checkpoint
    else:
        starting_epoch = 0
        n_iter = 0
        
    # switch to train mode
    model.train()

    # train model
    best_loss = float('inf')
    early_stopping = 0
    loss_margin = 0
    epochs_losses = []
    
    start_time = time.time()
    
    try:
        for epoch in tqdm(range(starting_epoch, EPOCHS), initial=starting_epoch, total=EPOCHS, desc="Epoch"):
            epoch_loss = 0
            
            for batch_idx, data in enumerate(train_dataloader):
                images = data['image'].to(device)
                        
                loss = model(images)
                
                loss.backward()
                
                if ((batch_idx + 1) % GRAD_ACCUMULATION == 0) or (batch_idx + 1 == len(train_dataloader)):
                    optimizer.step()
                    optimizer.zero_grad()
                    n_iter += 1
                    
                    # Safely save the last model weights and optimizer state
                    state = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'iteration': n_iter
                    }
                    safe_save(state, f'{save_model_path}/last_model.pth')
                    del state            

                # Update iteration
                loss = loss / GRAD_ACCUMULATION  # Normalize loss to account for batch accumulation
                epoch_loss += loss.item()
            
                # Save resnet backbone model every 2k iterations
                if n_iter % 2000 == 0 and n_iter!= 0 and batch_idx % GRAD_ACCUMULATION == 0:
                    print(f"Saving model weights at epoch {epoch} and iteration {n_iter}")
                    logging.info(f"\t\tEPOCH: {epoch} | ITER: {n_iter} | LOSS: {loss.item():.4f}")
                    # Save the model weights and optimizer state
                    torch.save(model.backbone.state_dict(), f'{save_model_path}/model_epoch{epoch}_iter{n_iter}.pth')
                                    
                    if n_iter % 20000 == 0:
                        print(f"Reaching 20k iterations... exiting training of model {SSL_METHOD} with backbone {BACKBONE_NAME}")
                        torch.save(model.backbone.state_dict(), f'{save_model_path}/model_epoch{epoch}_iter{n_iter}.pth')
                        raise StopIteration
            
            # Average loss for the epoch
            avg_loss = epoch_loss / len(train_dataloader)
            epochs_losses.append(avg_loss)
            
            # Update tqdm description every 10 epochs
            if epoch % 10 == 0:
                tqdm.write(f"Epoch {epoch}, Iteration: {n_iter}, Loss: {avg_loss:.4f}")

    # Catch keyboard interrupt, the StopIteration when reaching 20k iterations, and if the process is killed, and if there is any other exception
    except (KeyboardInterrupt, StopIteration, SystemExit, Exception) as e:
        print("\nTraining interrupted. Saving final state...")
        state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'iteration': n_iter
        }
        safe_save(state, f'{save_model_path}/last_model.pth')
        print(f"Final state saved in {save_model_path}/last_model.pth - Exiting...")

    finally:
        print(f"Training completed in {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
