

# ------------------------------------------------------------------------
#                               Libraries
# ------------------------------------------------------------------------

# General libraries
import os
import sys
import random
from datetime import datetime
import time
import argparse
import json

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
        default="downstream_task/config/config.json",
        help="Path to the JSON config file."
    )
    parser.add_argument(
        "-p",
        "--load_path",
        type=str,
        default=None,
        help="Path to the model to be loaded."
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
     
    # -------------------------------------------- PATHS -------------    
    PREFIX = generate_path(config["experiment_path"])
    log_file = f"{PREFIX}/experiments_results.txt"
    DATASET_NAME = config["dataset"]["name"]
    DATASET_PATH = os.path.join(config["dataset"]["path"], DATASET_NAME)
    
    # -------------------------------------------- PARAMETERS -------------
    # Dataset parameters
    SIZE = tuple(config["dataset"]["image_size"])
    NUM_CHANNELS = config["dataset"]["image_channels"]
    SIGMA = config["dataset"]["sigma"]
    TRAINING_SAMPLES = config["dataset"]["training_samples"]
    PIN_MEMORY = config["dataset"]["pin_memory"]
    NUM_WORKERS = 2 if config["dataset"]["num_workers"] == None else config["dataset"]["num_workers"]
    
    # Model parameters
    MODEL_NAME = config["model"]["name"]
    SSL_MODELS = ["moco", "mocov2", "mocov3", "simclr", "simclrv2", "dino", "barlow_twins", "byol"]
        
    if MODEL_NAME == "imagenet":
        MODEL_NAME = "smpUnet"
    elif MODEL_NAME == "ddpm":
        pass
    elif MODEL_NAME in SSL_MODELS:
        NUM_CHANNELS = 3    
    else:
        raise Exception("Model not found... Choose between: ddpm, imagenet, moco, mocov2, mocov3, simclr, simclrv2, dino, barlow_twins, byol")
    
    BACKBONE_NAME = config["model"]["encoder"]
    # Replace "efficientnet_b0" by "efficientnet-b0" and so on to match the model name
    BACKBONE_NAME = BACKBONE_NAME.replace("_", "-") if "efficientnet" in BACKBONE_NAME else BACKBONE_NAME
        
    PRETRAINED = config["training_protocol"]["scratch"]["apply"] == False
    NUM_EPOCHS = config["model"]["epochs"]
    BATCH_SIZE = config["dataset"]["batch_size"]
    GRAD_ACC = config["dataset"]["grad_accumulation"]
    LR = config["model"]["lr"] if PRETRAINED else config["model"]["lr"] / 0.1
    OPTIMIZER = config["model"]["optimizer"]
    SCHEDULER = config["model"]["scheduler"]
    LOSS_FUNCTION = config["model"]["loss_function"]
    PATIENCE = GRAD_ACC + 5
    EARLY_STOPPING = PATIENCE * 2 + 1
    print(f"Pretrained: {PRETRAINED} -> the actual learning rate is {LR}")
    
    # ---------------------------------------------------------------- DATASET ---------
    if DATASET_NAME == "chest":
        train_dataset = Chest(prefix=DATASET_PATH, phase='train', size=SIZE, num_channels=NUM_CHANNELS, sigma=SIGMA)
        val_dataset = Chest(prefix=DATASET_PATH, phase='validate', size=SIZE, num_channels=NUM_CHANNELS, sigma=SIGMA)
        test_dataset = Chest(prefix=DATASET_PATH, phase='test', size=SIZE, num_channels=NUM_CHANNELS, sigma=SIGMA)
    
    elif DATASET_NAME == "hand":
        train_dataset = Hand(prefix=DATASET_PATH, phase='train', size=SIZE, num_channels=NUM_CHANNELS, sigma=SIGMA)
        val_dataset = Hand(prefix=DATASET_PATH, phase='validate', size=SIZE, num_channels=NUM_CHANNELS, sigma=SIGMA)
        test_dataset = Hand(prefix=DATASET_PATH, phase='test', size=SIZE, num_channels=NUM_CHANNELS, sigma=SIGMA)
    elif DATASET_NAME == "cephalo":
        train_dataset = Cephalo(prefix=DATASET_PATH, phase='train', size=SIZE, num_channels=NUM_CHANNELS, sigma=SIGMA)
        val_dataset = Cephalo(prefix=DATASET_PATH, phase='validate', size=SIZE, num_channels=NUM_CHANNELS, sigma=SIGMA)
        test_dataset = Cephalo(prefix=DATASET_PATH, phase='test', size=SIZE, num_channels=NUM_CHANNELS, sigma=SIGMA)
    else:
        raise Exception("Dataset not found")

    NUM_LANDMARKS = train_dataset.num_landmarks
    
    # ---------------------------------------------------------------- DATA LOADING ---------
    # Randomly exclude images to reduce the number of samples in the training dataset
    #random_indices = np.random.choice(len(train_dataset), TRAINING_SAMPLES, replace=False)
    #print(random_indices)
    #train_dataset.indexes = [train_dataset.indexes[i] for i in sorted(random_indices)]
    
    if TRAINING_SAMPLES == "all":
        pass
    else:
        assert len(train_dataset) >= int(TRAINING_SAMPLES), "The number of training samples is greater than the number of samples in the dataset"
        
        train_dataset.indexes = train_dataset.indexes[:int(TRAINING_SAMPLES)]

    # create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS, drop_last=False)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,  pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)

    # ---------------------------------------------------------------- LOG FILE ---------
    # Print dataset and experiment info in log file
    res_file = open(log_file, 'a')
    print(f"\n\n\n {datetime.now()} ---------------------- {DATASET_NAME} -------------------------------------------", file=res_file)
    print(f"SIZE: {SIZE} | BATCH: {BATCH_SIZE} | GRAD ACC: {GRAD_ACC} | SIGMA: {SIGMA} | LR: {LR} | CHANNELS: {NUM_CHANNELS} | Train Samples {TRAINING_SAMPLES}", file=res_file)
    print(f"samples -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}", file=res_file)
    print(f"dataloaders -> Train: {len(train_dataloader)} | Val: {len(val_dataloader)} | Test: {len(test_dataloader)}", file=res_file)
    res_file.close()

    print(f"\n\n\n {datetime.now()} ---------------------- {DATASET_NAME} -------------------------------------------")
    print(f"SIZE: {SIZE} | BATCH: {BATCH_SIZE} | GRAD ACC: {GRAD_ACC} | SIGMA: {SIGMA} | LR: {LR} | CHANNELS: {NUM_CHANNELS} | Train Samples {TRAINING_SAMPLES}")
    print(f"samples -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print(f"dataloaders -> Train: {len(train_dataloader)} | Val: {len(val_dataloader)} | Test: {len(test_dataloader)}")
    # ---------------------------------------------------------------- MODEL ---------
    
    if MODEL_NAME == "smpUnet" and BACKBONE_NAME is not None:
        if PRETRAINED == True and config["training_protocol"]["finetuning"]["resume"] == False:
            model = smpUnet(
                encoder_name=BACKBONE_NAME,
                encoder_weights="imagenet",
                in_channels=NUM_CHANNELS,
                classes=NUM_LANDMARKS
            ).to(device)
            model_name = f"{MODEL_NAME}/{model.encoder_name}/{model.encoder_weights}"
        else:
            model = smpUnet(
                encoder_name=BACKBONE_NAME,
                encoder_weights=None,
                in_channels=NUM_CHANNELS,
                classes=NUM_LANDMARKS
            ).to(device)
            model_name = f"{MODEL_NAME}/{model.encoder_name}/random"
            
    elif MODEL_NAME in SSL_MODELS and BACKBONE_NAME is not None:
        model = smpUnet(
            encoder_name=BACKBONE_NAME,
            encoder_weights=None,
            in_channels=NUM_CHANNELS,
            classes=NUM_LANDMARKS
        ).to(device)
        
        assert os.path.exists(f'{config["training_protocol"]["finetuning"]["path"]}'), f"{BACKBONE_NAME} pretrained model path not found"
        
        model.encoder.load_state_dict(torch.load(f'{config["training_protocol"]["finetuning"]["path"]}', map_location=device))
        model_name = f"{MODEL_NAME}/{model.encoder_name}"
        
        
    elif MODEL_NAME == "ddpm":
        BACKBONE_NAME = ""
        model = Unet(
            dim=SIZE[0],
            channels=NUM_CHANNELS,
            dim_mults=[1,2,4,8],
            self_condition=True,
            resnet_block_groups=4,
            att_heads=4,
            att_res=32
        ).to(device)
            
        if PRETRAINED == True and config["training_protocol"]["finetuning"]["resume"] == False:
            model_name = f"{MODEL_NAME}/pretrained"            
            checkpoint = torch.load(config["training_protocol"]["finetuning"]["path"], map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            pretrained_epoch = checkpoint.get("epoch", "undefined")
            #print(f"Loaded model weights from {checkpoint['epoch']} epoch with fid {checkpoint['fid']}")
            del checkpoint
            """
            # freeze downsampling layers
            for name, param in model.named_parameters():
                if 'downs' in name:
                    param.requires_grad = False
            """
        else:
            model_name = f"{MODEL_NAME}/random"

        # change the number of output channels of the final convolutional layer
        model.final_conv = nn.Conv2d(model.final_conv.in_channels, NUM_LANDMARKS, 1)
    
    # ---------------------------------------------------------------- COUNT PARAMS ---------
    table, total_params = count_parameters(model)
    res_file = open(log_file, 'a')
    #print(table, file=res_file)
    print(f"Total Trainable Params: {total_params}", file=res_file)
    res_file.close()

    # ---------------------------------------------------------------- LOSS FUNCTION ---------
    if LOSS_FUNCTION == "CrossEntropyLoss":
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise Exception("Loss function not found... Choose between: CrossEntropyLoss")

    # ---------------------------------------------------------------- OPTIMIZER ---------  
    if OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=LR)
    elif OPTIMIZER == "AdamW":
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR)
    else:
        raise Exception("Optimizer not found... Choose between: Adam, AdamW")
        
    # ---------------------------------------------------------------- SCHEDULER ---------
    if SCHEDULER == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, patience=PATIENCE, factor=0.5, verbose=True)
    else:
        raise Exception("Scheduler not found... Choose between: ReduceLROnPlateau")


    # ---------------------------------------------------------------- MODEL PATHS ---------
    save_model_path = f"{PREFIX}/{DATASET_NAME}/size{SIZE[0]}x{SIZE[1]}_ch{NUM_CHANNELS}_samples{TRAINING_SAMPLES}/{model_name}"

    use_validation_set_for_inference = True if config["inference_protocol"]["use_validation_set_for_inference"]=="true" else False
    
    if use_validation_set_for_inference==True and PRETRAINED == True and config["model"]["name"] == "ddpm" and config["training_protocol"]["finetuning"]["resume"] == False:
        save_model_path = f"{save_model_path}/val/epoch{pretrained_epoch}"
    
    print(save_model_path)
    save_model_path = generate_path(save_model_path)

    load_model_path = os.path.join(save_model_path, f"best_checkpoint.pt")

    # ---------------------------------------------------------------- TRAINING ---------
    start_time = time.time()

    if config["training_protocol"]["apply"] == True:

        # Assert if the model is being trained from scratch or if it is being fine-tuned
        assert config["training_protocol"]["scratch"]["apply"] != config["training_protocol"]["finetuning"]["apply"], "Choose only one training protocol (scratch or finetuning)"
        print(f"Training model on the {'validation' if use_validation_set_for_inference==True else 'test'} dataset")
        
        # Get the training protocol
        if config["training_protocol"]["scratch"]["apply"] == True:
            loss_results = train_and_validate(model, device, train_dataloader, val_dataloader, optimizer, scheduler, loss_fn, NUM_EPOCHS, 
                                                save_model_path, patience=EARLY_STOPPING, useGradAcc=GRAD_ACC, continue_training=config["training_protocol"]["scratch"]["resume"])
        elif config["training_protocol"]["finetuning"]["apply"] == True:
            
            DIFFERENT_DATASET = True if config["training_protocol"]["finetuning"]["different_dataset"] == "true" else False
            
            if DIFFERENT_DATASET == True:
                load_path = config["training_protocol"]["finetuning"]["path"]
                assert os.path.exists(load_path), "Pretrained model path not found"
                loss_results = fine_tune(model, device, train_dataloader, val_dataloader, optimizer, scheduler, loss_fn, NUM_EPOCHS, 
                                                    load_path, save_model_path, patience=EARLY_STOPPING, useGradAcc=GRAD_ACC)
            else: 
                loss_results = train_and_validate(model, device, train_dataloader, val_dataloader, optimizer, scheduler, loss_fn, NUM_EPOCHS, 
                                                    save_model_path, patience=EARLY_STOPPING, useGradAcc=GRAD_ACC, continue_training=config["training_protocol"]["finetuning"]["resume"])       
        else:
            raise Exception("Training protocol not found... Choose between: scratch, finetuning")

    # ---------------------------------------------------------------- TESTING --------
    end_time = time.time()

    if args.load_path is not None:
        load_model_path = args.load_path
        
    if config["inference_protocol"]["apply"] == True:
        print(f"Testing model on the {'validation' if use_validation_set_for_inference==True else 'test'} dataset")
        res_file = open(log_file, 'a')
        print(f"Testing model on the {'validation' if use_validation_set_for_inference==True else 'test'} dataset", file=res_file)
        res_file.close()
        
        if use_validation_set_for_inference == True:
            test_loss, results, mre, sdr, mse, mAP_heatmaps, mAP_keypoints, iou, epoch  = evaluate(model, device, val_dataloader, loss_fn, load_model_path, 
                                            NUM_LANDMARKS, sigma=SIGMA, res_file_path=log_file)
        else:
            test_loss, results, mre, sdr, mse, mAP_heatmaps, mAP_keypoints, iou, epoch = evaluate(model, device, test_dataloader, loss_fn, load_model_path, 
                                            NUM_LANDMARKS, sigma=SIGMA, res_file_path=log_file)

    # ---------------------------------------------------------------- TELEGRAM ---------
    # Free GPU cache and RAM memory
    #free_gpu_cache()
    

    sdr_str = '\n'.join(f'\tThresholds {k}: {v*100:.2f}' for k, v in sorted(sdr.items()))

    message = (
        f"<b>{DATASET_NAME}</b> | Train Samples: {TRAINING_SAMPLES} \n"
        f"<b>Model:</b> {model_name} \n"
        f"<b>Shape:</b>[{SIZE}, {SIZE}, {NUM_CHANNELS}] \n"
        f"<b>Sigma:</b> {SIGMA} \n"
        f"<b>Batch:</b> {BATCH_SIZE}x{GRAD_ACC} \n"
        f"<b>Time:</b> {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))} \n" 
        f"<b>MRE:</b> {mre:.2f} \n\n"
        f"<b>SDR:</b> \n{sdr_str} \n"   
    )
    
    send_telegram_message(message)
    
    # Save the results in a file
    results_dir = f"outputs/{DATASET_NAME}_{MODEL_NAME}"
    os.makedirs(f'{results_dir}', exist_ok=True)
    
    if not os.path.exists(f'{results_dir}/outputs_{DATASET_NAME}_{MODEL_NAME}_{BACKBONE_NAME}_{TRAINING_SAMPLES}.txt'):
        with open(f'{results_dir}/outputs_{DATASET_NAME}_{MODEL_NAME}_{BACKBONE_NAME}_{TRAINING_SAMPLES}.txt', 'w') as f:
            print(f"\n\n{DATASET_NAME} | {MODEL_NAME} | {BACKBONE_NAME} | {TRAINING_SAMPLES}", file=f)
            print(f"Shape: [{SIZE}, {SIZE}, {NUM_CHANNELS}] | Sigma: {SIGMA} | Batch: {BATCH_SIZE}x{GRAD_ACC}", file=f)
            print(f"Time: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}", file=f)
            print(f"MRE: {mre:.2f}", file=f)
            print(f"SDR: \n{sdr_str}", file=f)
            print(f"MSE: {mse:.2f}", file=f)
            print(f"IOU: {iou:.2f}", file=f)
            print(f"mAP Heatmaps: {mAP_heatmaps:.2f}", file=f)
            print(f"mAP Keypoints: {mAP_keypoints:.2f}", file=f)
            print(f"Epoch: {epoch}", file=f)
            print(f"Test Loss: {test_loss:.2f}", file=f)
            print(f"Total Trainable Params: {total_params}", file=f)
            print(f"Model Path: {save_model_path}", file=f)
    else:
        with open(f'{results_dir}/outputs_{DATASET_NAME}_{MODEL_NAME}_{BACKBONE_NAME}_{TRAINING_SAMPLES}.txt', 'a') as f:
            print(f"\n\n{DATASET_NAME} | {MODEL_NAME} | {BACKBONE_NAME} | {TRAINING_SAMPLES}", file=f)
            print(f"Shape: [{SIZE}, {SIZE}, {NUM_CHANNELS}] | Sigma: {SIGMA} | Batch: {BATCH_SIZE}x{GRAD_ACC}", file=f)
            print(f"Time: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}", file=f)
            print(f"MRE: {mre:.2f}", file=f)
            print(f"SDR: \n{sdr_str}", file=f)
            print(f"MSE: {mse:.2f}", file=f)
            print(f"IOU: {iou:.2f}", file=f)
            print(f"mAP Heatmaps: {mAP_heatmaps:.2f}", file=f)
            print(f"mAP Keypoints: {mAP_keypoints:.2f}", file=f)
            print(f"Epoch: {epoch}", file=f)
            print(f"Test Loss: {test_loss:.2f}", file=f)
            print(f"Total Trainable Params: {total_params}", file=f)
            print(f"Model Path: {save_model_path}", file=f)
        
        
    