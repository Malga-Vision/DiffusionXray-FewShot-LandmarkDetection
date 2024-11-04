# ------------------------------------------------------------------------
#                               Libraries
# ------------------------------------------------------------------------

# General libraries
import os
import random
from datetime import datetime

# Deep learning libraries
import torch
from torch import nn 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Custom libraries
from utilities import *
from landmarks_datasets import * 
from model.deep_learning import *
from model.models import *

# Set random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42) 
torch.cuda.manual_seed(42)


import ssl

ssl._DEFAULT_CIPHERS = 'HIGH:!DH:!aNULL'

def ignore_ssl_certificate_verification():
    try:
        # Python 3.4+
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
    except AttributeError:
        # Python 2.x
        import requests
        from urllib3.exceptions import InsecureRequestWarning
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

ignore_ssl_certificate_verification()


## -----------------------------------------------------------------------------------------------------------------##
##                                      DATASETS                                             ##
## -----------------------------------------------------------------------------------------------------------------##

datasets_list = ["chest", "cephalo", "hand"] 
backbone_list = ["vgg19", "densenet161", "resnext50_32x4d"] #"efficientnet-b5"] 

EXPERIMENT_PATH = "downstream_task/landmarks_experiments/backbone_selection"

# Create folder for saving models 
if not os.path.exists(EXPERIMENT_PATH):
    os.makedirs(EXPERIMENT_PATH)

log_file = f"{EXPERIMENT_PATH}/experiments_results.txt"

NUM_EPOCHS = 200
K_FOLDS = 5
BATCH_SIZE = 2
GRAD_ACC = 8

LR = 1e-5
SIZE = (256, 256)
SIGMA = 5

PATIENCE = GRAD_ACC + 5
EARLY_STOPPING = PATIENCE * 2 + 1

NUM_CHANNELS = 1
ONLY_INFERENCE = False
PIN_MEMORY = True
NUM_WORKERS = os.cpu_count()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- CHEST ---------
CHEST_DATASET_PATH = 'datasets/chest'
assert os.path.exists(CHEST_DATASET_PATH), f"Chest dataset path does not exist: {CHEST_DATASET_PATH}, current path: {os.getcwd()}"
CHEST_NUM_LANDMARKS = 6

CHEST_SIZE = SIZE
CHEST_SIGMA = SIGMA

chest_train_dataset = Chest(prefix=CHEST_DATASET_PATH, phase='train', size=SIZE, num_channels=NUM_CHANNELS, sigma=SIGMA)
chest_val_dataset = Chest(prefix=CHEST_DATASET_PATH, phase='validate', size=SIZE, num_channels=NUM_CHANNELS, sigma=SIGMA)
chest_test_dataset = Chest(prefix=CHEST_DATASET_PATH, phase='test', size=SIZE, num_channels=NUM_CHANNELS, sigma=SIGMA)

chest_train_val_dataset = torch.utils.data.ConcatDataset([chest_train_dataset, chest_val_dataset])
print(f"CHEST: {len(chest_train_dataset)} | {len(chest_val_dataset)} | {len(chest_test_dataset)}")


# ---------------------------------------------------------------- CEPHALOMETRIC ---------
CEPHALOMETRIC_DATASET_PATH = 'datasets/cephalo'
assert os.path.exists(CEPHALOMETRIC_DATASET_PATH), f"Cephalometric dataset path does not exist: {CEPHALOMETRIC_DATASET_PATH}, current path: {os.getcwd()}"
CEPHALOMETRIC_NUM_LANDMARKS = 19

CEPHALOMETRIC_SIZE = SIZE
CEPHALOMETRIC_SIGMA = SIGMA

cephalo_train_dataset = Cephalo(prefix=CEPHALOMETRIC_DATASET_PATH, phase='train', size=SIZE, num_channels=NUM_CHANNELS, sigma=SIGMA)
cephalo_val_dataset = Cephalo(prefix=CEPHALOMETRIC_DATASET_PATH, phase='validate', size=SIZE, num_channels=NUM_CHANNELS, sigma=SIGMA)
cephalo_test_dataset = Cephalo(prefix=CEPHALOMETRIC_DATASET_PATH, phase='test', size=SIZE, num_channels=NUM_CHANNELS, sigma=SIGMA)

cephalo_train_val_dataset = torch.utils.data.ConcatDataset([cephalo_train_dataset, cephalo_val_dataset])
print(f"CEPHALO: {len(cephalo_train_dataset)} | {len(cephalo_val_dataset)} | {len(cephalo_test_dataset)}")


# ---------------------------------------------------------------- HAND ---------
HAND_DATASET_PATH = 'datasets/hand'
assert os.path.exists(HAND_DATASET_PATH), f"Hand dataset path does not exist: {HAND_DATASET_PATH}, current path: {os.getcwd()}"
HAND_NUM_LANDMARKS = 37

HAND_SIZE = SIZE
HAND_SIGMA =  SIGMA

hand_train_dataset = Hand(prefix=HAND_DATASET_PATH, phase='train', size=SIZE, num_channels=NUM_CHANNELS, sigma=SIGMA)
hand_val_dataset = Hand(prefix=HAND_DATASET_PATH, phase='validate', size=SIZE, num_channels=NUM_CHANNELS, sigma=SIGMA)
hand_test_dataset = Hand(prefix=HAND_DATASET_PATH, phase='test', size=SIZE, num_channels=NUM_CHANNELS, sigma=SIGMA)
    
hand_train_val_dataset = torch.utils.data.ConcatDataset([hand_train_dataset, hand_val_dataset])
print(f"HAND: {len(hand_train_dataset)} | {len(hand_val_dataset)} | {len(hand_test_dataset)}")

## -----------------------------------------------------------------------------------------------------------------##
##                                       TRAINING                                             ##
## -----------------------------------------------------------------------------------------------------------------##



for i in datasets_list:
    print(f"\n\n\n {datetime.now()} ---------------------- {i.upper()} -------------------------------------------")
    print(f"SIZE: {SIZE} | BATCH: {BATCH_SIZE} | GRAD ACC: {GRAD_ACC} | SIGMA: {SIGMA} | LR: {LR} | CHANNELS: {NUM_CHANNELS}")

    if i == "chest":
        NUM_LANDMARKS = CHEST_NUM_LANDMARKS
        dataset_name = i
        training_dataset = chest_train_val_dataset
        
    elif i == "hand":
        NUM_LANDMARKS = HAND_NUM_LANDMARKS
        dataset_name = i
        training_dataset = hand_train_val_dataset
    
    elif i == "cephalo":
        NUM_LANDMARKS = CEPHALOMETRIC_NUM_LANDMARKS
        dataset_name = i
        training_dataset = cephalo_train_val_dataset
    
    res_file = open(log_file, 'a')
    print(f"\n\n ----------------------------------------- {dataset_name.upper()} DATASET  ------------------------", file=res_file)
    res_file.close()

   
    # -------------------------------------------- SEGMENTATION MODELS -------------
    useHEATMAPS = True
    pretrained = "imagenet"

    for backbone in backbone_list:
                
        model = smpUnet(
            encoder_name=backbone,
            encoder_weights="imagenet",
            in_channels=NUM_CHANNELS,
            classes=NUM_LANDMARKS
        ).to(device)
        
        model_name = model.__class__.__name__

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR)
        scheduler = ReduceLROnPlateau(optimizer, patience=PATIENCE, factor=0.5) 

        res_file = open(log_file, 'a')
        print(f"\n\n --------- Model: {model_name}_{backbone} | Dataset: {dataset_name} | Batch: {BATCH_SIZE} | Sigma: {SIGMA} | Size: {SIZE}", file=res_file)
        res_file.close()

        save_model_path = generate_save_model_path(EXPERIMENT_PATH, model_name, dataset_name, SIGMA, SIZE, pretrained, backbone)

        k_fold_train_and_validate(model, device, training_dataset, optimizer, scheduler, loss_fn, NUM_EPOCHS, EARLY_STOPPING, BATCH_SIZE, GRAD_ACC, 
                                NUM_LANDMARKS, SIGMA, save_model_path, log_file, K_FOLDS, onlyInference=ONLY_INFERENCE)

        free_gpu_cache()
        del model, loss_fn, optimizer, scheduler