
import os
import numpy as np
from timeit import default_timer as timer
from tqdm.auto import tqdm 
import torch
import metrics
from torch import nn
import utilities
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
## -----------------------------------------------------------------------------------------------------------------##
##                                               TRAINING with GRADIENT ACCUMULATION                                                      ##
## -----------------------------------------------------------------------------------------------------------------##

def train_step(model: torch.nn.Module,
               device: torch.device,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               useHeatmaps: bool = False,
               gradient_accumulation_steps: int = 1):
    # Put model in train mode
    model = model.to(device)
    model.train()

    # Setup train loss value
    train_loss = 0.0

    # Loop through data loader data batches
    for batch, data in enumerate(dataloader):

        img_name = data['name']
        images_tensor = data['image']
        landmarks_tensor = data['landmarks']
        heatmaps_tensor = data['heatmaps']

        # Send data to target device
        X = images_tensor.to(device)

        if useHeatmaps:
            y = heatmaps_tensor.to(device)
        else:
            y = landmarks_tensor.to(device)

        #print(f"Batch {batch} - image tensor:  {X.shape} - GT tensor: {y.shape}")

        # Forward pass
        y_pred = model(X)

        #print(f"y pred shape: {y_pred.shape} - y shape: {y.shape}")
        
        # Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)

        # normalize loss to account for batch accumulation
        loss = loss / gradient_accumulation_steps
        train_loss += loss.item()

        # Loss backward
        loss.backward()
        
        # Check if it is time to update the weights
        if ((batch + 1) % gradient_accumulation_steps == 0) or (batch + 1 == len(dataloader)):
            # Optimizer step
            optimizer.step()
            # Reset gradients
            optimizer.zero_grad()

    # Adjust metrics to get average loss and accuracy per batch
    train_loss /= len(dataloader)
    
    return train_loss

## -----------------------------------------------------------------------------------------------------------------##
##                                               VALIDATION PART                                                    ##
## -----------------------------------------------------------------------------------------------------------------##

def validate_step(model: torch.nn.Module,
                  device: torch.device,
                  dataloader: torch.utils.data.DataLoader,
                  loss_fn: torch.nn.Module,
                  useHeatmaps: bool = False):
    # Put model in eval mode
    model = model.to(device)
    model.eval()

    # Setup validation loss value
    val_loss = 0.0

    with torch.no_grad():
        # Loop through DataLoader batches
        for batch, data in enumerate(dataloader):
            images_tensor = data['image']
            landmarks_tensor = data['landmarks']
            heatmaps_tensor = data['heatmaps']

            # Send data to target device
            X = images_tensor.to(device)

            if useHeatmaps:
                y = heatmaps_tensor.to(device)
            else:
                y = landmarks_tensor.to(device)

            # Forward pass
            val_pred_logits = model(X)

            # Calculate and accumulate loss
            loss = loss_fn(val_pred_logits, y)
            val_loss += loss.item()

    # Adjust metrics to get average loss per batch
    val_loss = val_loss / len(dataloader)

    return val_loss


## -----------------------------------------------------------------------------------------------------------------##
##                                               EARLY STOPPING                                                     ##
## -----------------------------------------------------------------------------------------------------------------##

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, delta=0, save_path=not None, counter=0, best_val_loss=None):
        self.patience = patience
        self.counter = counter
        self.best_val_loss = best_val_loss
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = save_path

    def call(self, val_loss, model, optimizer, scheduler, loss_fn, results, epochs_without_improvement, epoch):

        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            save_model(self.path, model, optimizer, scheduler, loss_fn, results, epochs_without_improvement, self.best_val_loss, epoch, called_by_early_stopping=True)

        elif val_loss >= self.best_val_loss + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_val_loss = val_loss
            save_model(self.path, model, optimizer, scheduler, loss_fn, results, epochs_without_improvement, self.best_val_loss, epoch, called_by_early_stopping=True)
            self.counter = 0


## -----------------------------------------------------------------------------------------------------------------##
##                                           SAVE AND LOAD A MODEL                                            ##
## -----------------------------------------------------------------------------------------------------------------##
def save_model(save_path, model, optimizer, scheduler, loss_fn, results, epochs_without_improvement, best_val_loss, epoch, called_by_early_stopping=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if called_by_early_stopping:
        checkpoint_path = os.path.join(save_path, "best_checkpoint.pt")
    else:
        checkpoint_path = os.path.join(save_path, f"checkpoint_epoch{epoch}.pt")

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss_fn': loss_fn.state_dict(),
        'results': results,
        'epochs_without_improvement': epochs_without_improvement,
        'best_val_loss': best_val_loss,
        'epoch': epoch
    }, checkpoint_path)
    #print(f"Model saved to {checkpoint_path}")

    
def load_model(load_path, model, optimizer, scheduler, loss_fn, device):
    checkpoint = torch.load(load_path, map_location=torch.device(device))

    # Load the state_dict into the model only if it exists in the checkpoint
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)  # Move the model to the specified device

    # Load the optimizer state_dict only if it exists in the checkpoint
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load the scheduler state_dict only if it exists in the checkpoint
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Load the loss_fn state_dict only if it exists in the checkpoint
    if 'loss_fn' in checkpoint:
        loss_fn.load_state_dict(checkpoint['loss_fn'])

    # Load other values only if they exist in the checkpoint
    start_epoch = checkpoint.get('epoch', 0) + 1
    results = checkpoint.get('results', None)
    epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
    best_val_loss = checkpoint.get('best_val_loss', None)
    print(f"Model loaded from {load_path} | Starting from epoch {start_epoch} | Best validation loss: {best_val_loss} | Epochs without improvement: {epochs_without_improvement}")
    return model, optimizer, scheduler, loss_fn, start_epoch, results, epochs_without_improvement, best_val_loss


## -----------------------------------------------------------------------------------------------------------------##
##                                           TRAINING + VALIDATION PART                                             ##
## -----------------------------------------------------------------------------------------------------------------##

def train_and_validate(model: torch.nn.Module,
                       device: torch.device,
                       train_dataloader: torch.utils.data.DataLoader,
                       val_dataloader: torch.utils.data.DataLoader,
                       optimizer: torch.optim.Optimizer,
                       scheduler: torch.optim.lr_scheduler,
                       loss_fn: torch.nn.Module,
                       epochs: int = 10,
                       save_path: str = None,
                       useHeatmaps: bool = True,
                       patience: int = 10,
                       save_all_epochs: bool = False,
                       useGradAcc: int = 1,
                       continue_training: bool = False):
    
    if continue_training:
        model_path = os.path.join(save_path, "best_checkpoint.pt")
        assert model_path is not None, "If you want to continue training, you must provide a path to load the model from."

        # Load the model from the path
        model, optimizer, scheduler, loss_fn, start_epoch, results, epochs_without_improvement, best_val_loss = load_model(model_path, model, optimizer, scheduler, loss_fn, device)
    else:
        # Create empty results dictionary and initialize epoch
        results = {"train_loss": [], "val_loss": []}
        start_epoch = 1
        best_val_loss = float("inf")
        epochs_without_improvement = 0

    # Start the timer
    start_time = timer()

    # Create EarlyStopping instance
    early_stopping = EarlyStopping(patience=patience, save_path=save_path, counter=epochs_without_improvement, best_val_loss=best_val_loss)

    # Loop through training and validating steps for a number of epochs
    for epoch in tqdm(range(start_epoch, epochs + 1)):

        assert useGradAcc >= 1, "Gradient accumulation steps must be greater than 1"

        train_loss = train_step(model, device, train_dataloader, loss_fn, optimizer, useHeatmaps, gradient_accumulation_steps=useGradAcc)

        val_loss = validate_step(model, device, val_dataloader, loss_fn, useHeatmaps)

        scheduler_type = scheduler.__class__.__name__
        if scheduler_type == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        else:
            # Update the learning rate using the scheduler
            scheduler.step()

        # Print out what's happening
        print(f"Epoch {epoch} | Train Loss: {train_loss:.7f} | Validation Loss: {val_loss:.7f}")

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)

        # Save the trained model
        if save_all_epochs is True:
            save_model(save_path, model, optimizer, scheduler, loss_fn, results, epochs_without_improvement, best_val_loss, epoch)

        # Check for early stopping
        early_stopping.call(val_loss, model, optimizer, scheduler, loss_fn, results, epochs_without_improvement, epoch)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
        
    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time: {end_time - start_time:.3f} seconds")

    # Return the filled results at the end of the epochs
    return results

## -----------------------------------------------------------------------------------------------------------------##
##                                                  FINE-TUNING IN-DOMAIN                                           ##
## -----------------------------------------------------------------------------------------------------------------##
def fine_tune(model: torch.nn.Module,
              device: torch.device,
              train_dataloader: torch.utils.data.DataLoader,
              val_dataloader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              scheduler: torch.optim.lr_scheduler,
              loss_fn: torch.nn.Module,
              epochs: int = 10,
              load_path: str = None,
              save_path: str = None,
              useHeatmaps: bool = True,
              patience: int = 10,
              useGradAcc: int = 1):
    
    assert load_path is not None, "You must provide a path to load the model from."
    
    # Load the model from the path
    model.load_state_dict(torch.load(load_path, map_location=torch.device(device)), strict=False) 
    model = model.to(device)  # Move the model to the specified device
    
    # Create empty results dictionary and initialize epoch
    results = {"train_loss": [], "val_loss": []}
    start_epoch = 1
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    
    # Start the timer
    start_time = timer()

    # Create EarlyStopping instance
    early_stopping = EarlyStopping(patience=patience, save_path=save_path, counter=epochs_without_improvement, best_val_loss=best_val_loss)

    # Loop through training and validating steps for a number of epochs
    for epoch in tqdm(range(start_epoch, epochs + 1)):

        assert useGradAcc >= 1, "Gradient accumulation steps must be greater than 1"

        train_loss = train_step(model, device, train_dataloader, loss_fn, optimizer, useHeatmaps, gradient_accumulation_steps=useGradAcc)

        val_loss = validate_step(model, device, val_dataloader, loss_fn, useHeatmaps)

        scheduler_type = scheduler.__class__.__name__
        if scheduler_type == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        else:
            # Update the learning rate using the scheduler
            scheduler.step()

        # Print out what's happening
        print(f"Epoch {epoch} | Train Loss: {train_loss:.7f} | Validation Loss: {val_loss:.7f}")

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)

        # Check for early stopping
        early_stopping.call(val_loss, model, optimizer, scheduler, loss_fn, results, epochs_without_improvement, epoch)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
        
    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time: {end_time - start_time:.3f} seconds")

    # Return the filled results at the end of the epochs
    return results   
                 


## -----------------------------------------------------------------------------------------------------------------##
##                                                  EVALUATION PART                                                 ##
## -----------------------------------------------------------------------------------------------------------------##
def test_step(model: torch.nn.Module,
              device: torch.device,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              num_landmarks: int,
              useHeatmaps: bool = False,
              sigma: int = 1.5,
              load_path: str = None):
    
    # Take the baseline of the path
    if load_path is not None:
        model_dir = os.path.dirname(load_path)
        
    # Put model in eval mode
    model = model.to(device)
    model.eval()
    model_name = model.__class__.__name__
    model_encoder = model.encoder.__class__.__name__ if hasattr(model, 'encoder') else ""

    # Setup test loss and test accuracy values
    test_loss = 0.0
    results = {}
    distances = [] 

    with torch.no_grad():
        # Loop through DataLoader batches
        for batch, data in enumerate(dataloader):
            images_name = data['name']
            images_tensor = data['image']
            #image_size = images_tensor.numpy().shape[2:]
            landmarks_tensor = data['landmarks']
            heatmaps_tensor = data['heatmaps']
            original_size = data['original_size']
            resized_size = data['resized_size']

            # Send data to target device
            X = images_tensor.to(device)

            if useHeatmaps:
                y = heatmaps_tensor.to(device)
            else:
                y = landmarks_tensor.to(device)

            # Forward pass
            y_pred = model(X)

            # Calculate and accumulate loss
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            # Move the prediction and the GT to the CPU
            y_pred = y_pred.cpu()

            # Save the prediction heatmaps as images in the model directory 
            #os.makedirs(f"{model_dir}/predictions", exist_ok=True)
            #utilities.save_heatmaps(X, y_pred, images_name, f"{model_dir}/predictions")                

            # Compute the MSE and mAP between the original landmarks and the predicted landmarks
            mse_list, mAP_list_heatmaps, mAP_list_keypoints, iou_list, distance_list = metrics.compute_batch_metrics(landmarks_tensor, heatmaps_tensor, y_pred, resized_size, num_landmarks, useHeatmaps, sigma)
            # Append to full list in order to compute the MRE and SDR for all the images
            distances.extend(distance_list)
            
            # Store image names as keys and their corresponding predictions as values.
            for i, name in enumerate(images_name):  # Since they are in batch I loop them
                # Storing prediction and metrics values
                results[name] = { 
                    'prediction': y_pred[i],
                    'mse': mse_list[i],
                    'map1': mAP_list_heatmaps[i],
                    'map2': mAP_list_keypoints[i],
                    'iou': iou_list[i]
                }

            del batch, data, images_name, images_tensor, landmarks_tensor, heatmaps_tensor, original_size, resized_size, X, y, y_pred, loss, mse_list, mAP_list_heatmaps, mAP_list_keypoints, iou_list, distance_list   # Free memory
                                       

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)

    # Compute metrics on full list
    #print("Dist shape:", len(distances))
    #print("Mean distance:", np.mean(distances))
    #print("Std distance:", np.std(distances))
    #print("Distances under 3px:", len([i for i in distances if i < 3]))
    #print("Distances above 15px:", len([i for i in distances if i > 15]))
    
    mre = metrics.compute_mre(distances)
    sdr = metrics.compute_sdr(distances)

    return test_loss, results, mre, sdr


def evaluate(model: torch.nn.Module,
          device: torch.device,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          load_path: str,
          num_landmarks: int = 6,
          useHeatmaps: bool = True,
          sigma: int = 1.5,
          currentKfold: int = 1,
          res_file_path: str = "results/readable_res.csv"):
    
    checkpoint = torch.load(load_path, map_location=torch.device(device))
    #model.load_state_dict(checkpoint['model'])

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nModel loaded from {load_path}")
    epoch = checkpoint.get('epoch', "Undefined")

    # Get the loss and the predictions dictionary
    test_loss, results, mre, sdr = test_step(model, device, test_dataloader, loss_fn, num_landmarks, useHeatmaps, sigma, load_path)

    total_mse_list = []
    total_mAP_heatmaps_list = []
    total_mAP_keypoints_list = []
    total_iou_list = []

    # Create a list with all metrics of all images
    for value in results.values():
        total_mse_list.append(value['mse'])
        total_mAP_heatmaps_list.append(value['map1'])
        total_mAP_keypoints_list.append(value['map2'])
        total_iou_list.append(value['iou'])

    # Compute the mean between all samples
    total_mse_mean = np.mean(total_mse_list)
    total_mAP_heatmaps_mean = np.mean(total_mAP_heatmaps_list)
    total_mAP_keypoints_mean = np.mean(total_mAP_keypoints_list)
    total_iou_mean = np.mean(total_iou_list)

    # Compute the standard deviation between all samples
    total_mse_std = np.std(total_mse_list)
    total_mAP_heatmaps_std = np.std(total_mAP_heatmaps_list)
    total_mAP_keypoints_std = np.std(total_mAP_keypoints_list)
    total_iou_std = np.std(total_iou_list)

    # Create a string representation of the sdr dictionary
    sdr_str = '\n'.join(f'\tThresholds {k}: {v*100:.2f}' for k, v in sorted(sdr.items()))

    # Print and Save results
    res_file = open(res_file_path, 'a')
    print(f"\n{load_path}", file=res_file)
    print(f"Fold {currentKfold} - Epoch: {epoch} | MSE: {total_mse_mean:.2f} ± {total_mse_std:.2f} | mAP heat: {total_mAP_heatmaps_mean:.2f} ± {total_mAP_heatmaps_std:.2f} | mAP key: {total_mAP_keypoints_mean:.2f} ± {total_mAP_keypoints_std:.2f} | IoU: {total_iou_mean:.2f} ± {total_iou_std:.2f} \nMRE: {mre:.2f} \nSDR: \n{sdr_str}", file=res_file)
    res_file.close()

    print(f"Fold {currentKfold} - Epoch: {epoch} | \nMSE: {total_mse_mean:.2f} ± {total_mse_std:.2f} | \nmAP heat: {total_mAP_heatmaps_mean:.2f} ± {total_mAP_heatmaps_std:.2f} | mAP key: {total_mAP_keypoints_mean:.2f} ± {total_mAP_keypoints_std:.2f} | \nIoU: {total_iou_mean:.2f} ± {total_iou_std:.2f} | \nMRE: {mre:.2f} | \nSDR: \n{sdr_str}")
    del total_mse_list, total_mAP_heatmaps_list, total_mAP_keypoints_list, total_iou_list

    return test_loss, results, mre, sdr, total_mse_mean, total_mAP_heatmaps_mean, total_mAP_keypoints_mean, total_iou_mean, epoch




# ------------------------------------------------------------------------
#                               Reinstantiate Model
# ------------------------------------------------------------------------

def reset_all_weights(model: nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)
     
def reinstantiate_model(model, optimizer, scheduler):
    model_type = model.__class__.__name__
    scheduler_type = scheduler.__class__.__name__
    optimizer_type = optimizer.__class__.__name__
    #print(scheduler_params)
    
    reset_all_weights(model)

    if optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=optimizer.param_groups[0]['lr'])
    else:    
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    if scheduler_type == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=scheduler.factor, patience=scheduler.patience, verbose=True, mode=scheduler.mode)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    return model, optimizer, scheduler


## -----------------------------------------------------------------------------------------------------------------##
##                                                            K-FOLD                                             ##
## -----------------------------------------------------------------------------------------------------------------##

        
def k_fold_train_and_validate(model: torch.nn.Module,
                                device: torch.device,
                                train_dataset: torch.utils.data.Dataset,
                                optimizer: torch.optim.Optimizer,
                                scheduler: torch.optim.lr_scheduler,
                                loss_fn: torch.nn.Module,
                                epochs: int,
                                early_stopping: int,
                                batch_size: int,
                                gradient_accumulation_steps: int,
                                num_landmarks: int,
                                sigma: int,
                                save_model_path: str,
                                log_file: str,
                                k_folds: int = 5,
                                onlyInference: bool = True
                                ):
    
    if onlyInference:
        k_train_losses = [0]
        k_val_losses = [0]
    else:
        k_train_losses = []
        k_val_losses = []

    k_test_losses = []
    k_mse = []
    k_iou = []
    k_map_heat = []
    k_map_key = []
    k_mre = []
    k_sdr = {}

    results_folds = []
        
    # Get the total number of samples
    total_size = len(train_dataset)

    # Divide by the number of folds to get the size of each fold
    fold_size = total_size // k_folds

    indices = list(range(total_size))
  

    for fold in range(k_folds):
        
        # Assign the fold as the val set
        val_ids = indices[fold*fold_size:(fold+1)*fold_size]

        # The remaining data will be used for training 
        train_ids = indices[:fold*fold_size] + indices[(fold+1)*fold_size:]

        # Create the subsets
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        # Create the data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler, num_workers=4, pin_memory=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=val_subsampler, num_workers=4, pin_memory=True)

        save_fold_path = f"{save_model_path}/fold_{fold}"
        print(f"Training fold {fold}...")
        print(f"Path: {save_fold_path}")
        
        if not onlyInference:
            
            model, optimizer, scheduler = reinstantiate_model(model, optimizer, scheduler)
            
            # Train on the current fold
            fold_train_results = train_and_validate(model, device, train_loader, val_loader, optimizer, scheduler, loss_fn, epochs, 
                                            save_fold_path, patience=early_stopping, useGradAcc=gradient_accumulation_steps, continue_training=False)
            
            last_train_loss = fold_train_results['train_loss'][-1]
            last_val_loss = fold_train_results['val_loss'][-1]

            k_train_losses.append(last_train_loss)
            k_val_losses.append(last_val_loss)

            print(f"FOLD {fold} | Train loss: {last_train_loss} | Val loss: {last_val_loss}")
            del fold_train_results, last_train_loss, last_val_loss, train_loader, train_subsampler, val_subsampler, train_ids, val_ids


        # ---------------------- Evaluate performances on val set (the training never has seen the images on the val set, it use only to minimize error) -------------------------------
        load_fold_path = os.path.join(save_fold_path, f"best_checkpoint.pt")
        # Get the loss and the predictions dictionary
        test_loss, results, mre, sdr, mse, mAP_heatmaps, mAP_keypoints, iou, epoch = evaluate(model, device, val_loader, loss_fn, load_fold_path, 
                                        num_landmarks, sigma, res_file_path=log_file)        

        k_test_losses.append(test_loss)
        
        k_mre.append(mre)
        
        # Update the sdr dictionary
        for threshold, value in sdr.items():
            if threshold not in k_sdr:
                k_sdr[threshold] = []
            k_sdr[threshold].append(value)

        # Create a list with all metrics of all images
        for value in results.values():
            k_mse.append(value['mse'])
            k_map_heat.append(value['map1'])
            k_map_key.append(value['map2'])
            k_iou.append(value['iou'])    

        del test_loss, results, mre, sdr, load_fold_path, val_loader, 
        
    # Compute the mean and SD for each threshold
    sdr_mean_std = {threshold: (np.mean(values), np.std(values)) for threshold, values in k_sdr.items()}

    # Compute the mean for the losses
    k_train_loss_mean = np.mean(k_train_losses)
    k_train_loss_std = np.std(k_train_losses)

    k_val_loss_mean = np.mean(k_val_losses)
    k_val_loss_std = np.std(k_val_losses)

    k_test_loss_mean = np.mean(k_test_losses)
    k_test_loss_std = np.std(k_test_losses)

    # Compute the mean between all samples
    k_mse_mean = np.mean(k_mse)
    k_map_heat_mean = np.mean(k_map_heat)
    k_map_key_mean = np.mean(k_map_key)
    k_iou_mean = np.mean(k_iou)

    # Compute the standard deviation between all samples
    k_mse_std = np.std(k_mse)
    k_map_heat_std = np.std(k_map_heat)
    k_map_key_std = np.std(k_map_key)
    k_iou_std = np.std(k_iou)

    # Compute the mean MRE and mean SDR
    k_mre_mean = np.mean(k_mre)
    k_mre_std = np.std(k_mre)

    res_file = open(log_file, 'a')
    print(f"----------------------------------------------------------------- GLOBAL RES for {k_folds} Folds \n",
        f"Train loss ---> Mean: {k_train_loss_mean} | Std: {k_train_loss_std} \n",
        f"Val loss ---> Mean: {k_val_loss_mean} | Std: {k_val_loss_std} \n",
        f"Test loss ---> Mean: {k_test_loss_mean} | Std: {k_test_loss_std} \n",
        f"MSE ---> Mean: {k_mse_mean:.2f} | Std: {k_mse_std:.2f} \n",
        f"mAp heat ---> Mean: {k_map_heat_mean:.2f} | Std: {k_map_heat_std:.2f} \n",
        f"mAp key ---> Mean: {k_map_key_mean:.2f} | Std: {k_map_key_std:.2f} \n",
        f"IOU ---> Mean: {k_iou_mean:.2f} | Std: {k_iou_std:.2f} \n",
        f"MRE ---> Mean: {k_mre_mean:.2f} | Std: {k_mre_std:.2f} \n",
        f"SDR:\n",
        *(f"Threshold {threshold}: Mean: {mean*100:.2f} | Std: {std*100:.2f}\n" for threshold, (mean, std) in sdr_mean_std.items()),
        file=res_file)
    res_file.close()


    print(f"----------------------------------------------------------------- GLOBAL RES for {k_folds} Folds \n",
        f"Train loss ---> Mean: {k_train_loss_mean} | Std: {k_train_loss_std} \n",
        f"Val loss ---> Mean: {k_val_loss_mean} | Std: {k_val_loss_std} \n",
        f"Test loss ---> Mean: {k_test_loss_mean} | Std: {k_test_loss_std} \n",
        f"MSE ---> Mean: {k_mse_mean:.2f} | Std: {k_mse_std:.2f} \n",
        f"mAp heat ---> Mean: {k_map_heat_mean:.2f} | Std: {k_map_heat_std:.2f} \n",
        f"mAp key ---> Mean: {k_map_key_mean:.2f} | Std: {k_map_key_std:.2f} \n",
        f"IOU ---> Mean: {k_iou_mean:.2f} | Std: {k_iou_std:.2f} \n",
        f"MRE ---> Mean: {k_mre_mean:.2f} | Std: {k_mre_std:.2f} \n",
        f"SDR:\n",
        *(f"Threshold {threshold}: Mean: {mean*100:.2f} | Std: {std*100:.2f}\n" for threshold, (mean, std) in sdr_mean_std.items()))
    del k_train_losses, k_val_losses, k_test_losses, k_mse, k_iou, k_map_heat, k_map_key, k_mre, k_sdr, results_folds, train_dataset, total_size, fold_size, indices    