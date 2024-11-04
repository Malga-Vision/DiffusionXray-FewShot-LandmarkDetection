
# ------------------------------------------------------------------------
#                               Libraries
# ------------------------------------------------------------------------

# General libraries
import os
import cv2
from matplotlib import pyplot as plt
from prettytable import PrettyTable

# Deep learning libraries
import torch
import torchvision
from torch.utils.data import DataLoader
import albumentations as A

# Custom libraries
from ddpm_datasets import ChestDiffusionDataset, HandDiffusionDataset, CephaloDiffusionDataset

# ------------------------------------------------------------------------
#                               Logging and Utilities
# ------------------------------------------------------------------------

# Generate path if it does not exist
def generate_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# Get the current GPU memory usage by tensors in megabytes for a given device
def gpu_memory_usage(device):
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    print(f'Allocated memory: {allocated / (1024 ** 2):.2f} MB')
    print(f'Reserved memory: {reserved / (1024 ** 2):.2f} MB')

# Compute the number of trainable parameters in a model
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    #print(table)
    print(f"Total Trainable Params: {total_params}")
    return table, total_params

# ------------------------------------------------------------------------
#                               Visualizations
# ------------------------------------------------------------------------
   
def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, title, image_names, **kwargs):
    batch_size = images[0].shape[0]
    fig, axs = plt.subplots(3, batch_size, figsize=(batch_size * 2, 6))
    
    for i in range(batch_size):
        axs[0, i].imshow(images[0][i].permute(1, 2, 0).to('cpu').numpy())
        axs[0, i].set_title(f"{image_names[i]}")
        axs[0, i].axis('off')
        
        axs[1, i].imshow(images[1][i].permute(1, 2, 0).to('cpu').numpy())
        axs[1, i].set_title("Reconstructed")
        axs[1, i].axis('off')
        
        axs[2, i].imshow(images[2][i].permute(1, 2, 0).to('cpu').numpy())
        axs[2, i].set_title("Difference")
        axs[2, i].axis('off')
    
    fig.suptitle(f"{title}")
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)

def check_pixels_range_of_image(tensor):
    # Ensure the input is a tensor
    assert torch.is_tensor(tensor), "Input must be a tensor"

    # Flatten the tensor to get all pixel values
    pixel_values = tensor.view(-1)

    # Compute min and max values
    min_val = pixel_values.min().item()
    max_val = pixel_values.max().item()

    #print(f"The range of pixel values is: {min_val} to {max_val}")
    return min_val, max_val


def compute_diff(x, x_hat):
    # Ensure both tensors are on the same device
    assert x.device == x_hat.device, "Tensors must be on the same device"
    
    # Ensure both tensors have the same shape
    assert x.shape == x_hat.shape, "Tensors must have the same shape"
    
    x_min, x_max = check_pixels_range_of_image(x)
    x_hat_min, x_hat_max = check_pixels_range_of_image(x_hat)

    # Ensure both tensors are have pixel values in the range [0, 1]
    #assert x_min >= 0 and x_max <= 1, f"Pixel values of x must be in the range [0, 1]. Actual range: [{x_min}, {x_max}]"
    #assert x_hat_min >= 0 and x_hat_max <= 1, f"Pixel values of x_hat must be in the range [0, 1]. Actual range: [{x_hat_min}, {x_hat_max}]"
    #print(f"Pixel values of x are in the range [{x_min}, {x_max}]")
    #print(f"Pixel values of x_hat are in the range [{x_hat_min}, {x_hat_max}]")
    # Compute absolute difference
    diff = torch.abs(x - x_hat)
    
    # Normalize to the range [0, 1] and return the difference image
    diff = (diff - diff.min()) / (diff.max() - diff.min())  

    return diff


# ------------------------------------------------------------------------
#                               Data Loading and Preprocessing
# ------------------------------------------------------------------------

def get_transforms(image_size, phase='train'):
    resize_image_size = int(image_size*1.02)
    if phase == 'train':
        return A.Compose([
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0, rotate_limit=2, border_mode=cv2.BORDER_REPLICATE, p=0.5),
        #A.Perspective(scale=(0, 0.02), pad_mode=cv2.BORDER_REPLICATE, p=0.5),
        A.Resize(image_size, image_size),
        #A.RandomCrop(height=image_size, width=image_size),
        #A.HorizontalFlip(p=1),
        #A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.2, 0.2), p=0.5),
        A.Normalize(normalization='min_max'),
        A.pytorch.ToTensorV2()
    ])

    elif phase == 'test':
        return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(normalization='min_max'),
        A.pytorch.transforms.ToTensorV2()
        ])
    else:
        raise ValueError('phase must be either "train" or "test"')


def load_data(dataset_path, image_size, image_channels, batch_size, pin_memory=False, num_workers = os.cpu_count()):    
    dataset_name = os.path.basename(dataset_path)
    
    transforms_train = get_transforms(image_size, phase='train')
    transforms_test = get_transforms(image_size, phase='test')
    
    if dataset_name == 'chest':
        train_dataset = ChestDiffusionDataset(dataset_path, channels=image_channels, transform=transforms_train, phase='train')
        test_dataset = ChestDiffusionDataset(dataset_path, channels=image_channels, transform=transforms_test, phase='test')
    elif dataset_name == 'hand':
        train_dataset = HandDiffusionDataset(dataset_path, channels=image_channels, transform=transforms_train, phase='train')
        test_dataset = HandDiffusionDataset(dataset_path, channels=image_channels, transform=transforms_test, phase='test')
    elif dataset_name == 'cephalo':
        train_dataset = CephaloDiffusionDataset(dataset_path, channels=image_channels, transform=transforms_train, phase='train')
        test_dataset = CephaloDiffusionDataset(dataset_path, channels=image_channels, transform=transforms_test, phase='test')
    else:
        raise ValueError('Dataset name must be either "chest" or "hand" or "cephalo"')
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    
    return train_dataloader, test_dataloader




