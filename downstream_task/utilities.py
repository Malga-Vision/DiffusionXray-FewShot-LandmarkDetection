import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import requests
import torch
from GPUtil import showUtilization as gpu_usage
from prettytable import PrettyTable

## -----------------------------------------------------------------------------------------------------------------##
##                                          CLEAN GPU MEMORY USAGE AND DATASETS                                     ##
## -----------------------------------------------------------------------------------------------------------------##

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage() 
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print("GPU Usage after emptying the cache")
    gpu_usage()

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

## -----------------------------------------------------------------------------------------------------------------##
##                            HEATMAPS GENERATION FROM LANDMARKS POINTS                                             ##
## -----------------------------------------------------------------------------------------------------------------##

def fuse_heatmaps(heatmaps):
    fused_heatmap = np.sum(heatmaps, axis=0)

    # Threshold the heatmap so that values below a certain threshold are set to 0
    binary_fused_heatmap = np.where(fused_heatmap < 0.5, 0, 1)

    assert is_binary_image(binary_fused_heatmap), "Image is not binary"

    return binary_fused_heatmap

def fuse_heatmaps(heatmaps):
    # Use np.maximum.reduce to get the maximum value from each stack of pixels across the heatmaps
    fused_heatmap = np.maximum.reduce(heatmaps)
    return fused_heatmap

def scale_points(points: list, img_size: tuple, orig_size: tuple = None, offset=0):    
    # Scale coordinates according to image size
    if orig_size:
        # Points * Ratio -> Downscaling
        scaled_points = [tuple([round(p*isize/osize)+offset for p, isize, osize in zip(point, img_size, orig_size)]) for point in points]
    else:
        # Points * current Size -> Upscaling (when i use "extract_landmarks" function the points are with size (1,1))
        scaled_points = [tuple([round(r*sz)+offset for sz, r in zip(point, img_size)]) for point in points]

    return scaled_points


def points_to_heatmap(points: list, img_size: tuple, orig_size: tuple = None, sigma=2, fuse=False, offset=0):

    # Scale coordinates according to image size
    if orig_size:
        scaled_points = scale_points(points, img_size, orig_size, offset=offset)
    else:
        scaled_points = scale_points(points, img_size, offset=offset)

    # Generate heatmaps with the scaled points
    heatmaps = [generate_heatmap_from_points(point, img_size, sigma) for point in scaled_points]

    if fuse:
        heatmaps = fuse_heatmaps(heatmaps)

    return np.array(heatmaps)


def generate_heatmap_from_points(point, img_size, sigma):
    # Create a meshgrid of x,y coordinates for the image size
    x, y = np.meshgrid(np.arange(img_size[0]), np.arange(img_size[1]))
    # Calculate the heatmap using a Gaussian function centered at the input point
    heatmap = np.exp(-((x - point[0]) ** 2 + (y - point[1]) ** 2) / (2 * sigma ** 2))
    # Threshold the heatmap so that values below a certain threshold are set to 0
    binary_heatmap = np.where(heatmap < 0.5*heatmap.max(), 0, 1)

    assert is_binary_image(binary_heatmap), "Image is not binary"

    return binary_heatmap

def is_binary_image(image):
    binary_check = np.logical_or(image == 0, image == 1)
    return binary_check.all()
## -----------------------------------------------------------------------------------------------------------------##
##                                   LANDMARKS EXTRACTION FROM HEATMAPS                                             ##
## -----------------------------------------------------------------------------------------------------------------##

def extract_landmarks(heatmaps, num_landmarks):
    landmarks = np.zeros((num_landmarks, 2), dtype=np.float64)
    heatmap_height, heatmap_width = heatmaps.shape[1], heatmaps.shape[2]

    # Loop all the heatmaps (one for each landmark)
    for i in range(num_landmarks):
        heatmap_channel = heatmaps[i]  # get the heatmap number i
        
        # binarize the heatmap channel using a threshold
        binary_img = np.where(heatmap_channel < 0.5 * heatmap_channel.max(), 0, 1)
        binary_img = binary_img.astype(np.uint8)  # convert the binary image to uint8 datatype

        assert is_binary_image(binary_img), "Image is not binary"

        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)  # find the contours in the binary image

        if contours:
            max_contour = max(contours, key=cv2.contourArea)  # find the contour with the maximum area
            M = cv2.moments(max_contour)  # calculate the moments of the maximum contour

            if M['m00'] != 0:  # avoid divide by zero error
                centroid_x = M['m10'] / M['m00']  # calculate the x-coordinate of centroid
                centroid_y = M['m01'] / M['m00']  # calculate the y-coordinate of centroid
                landmarks[i, :] = [centroid_x / heatmap_height,
                                   centroid_y / heatmap_width]  # normalize the coordinates

    return landmarks


## -----------------------------------------------------------------------------------------------------------------##
##                                                  PLOT LOSS CURVES                                                ##
## -----------------------------------------------------------------------------------------------------------------##

def plot_loss_curves(results_path: str, save_dir: str = None):
    # Load the results dictionary
    results = torch.load(results_path)['results']
    # Get the loss values of the results dictionary (training and validation)
    train_loss = results['train_loss']
    val_loss = results['val_loss']

    # Figure out how many epochs there were
    epochs = range(1, len(results['train_loss']) + 1)

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label='train_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f"loss_epochs{epochs}"))

    plt.show()

# -----------------------------------------------------------------------------------------------------------------##
# 
# -----------------------------------------------------------------------------------------------------------------##

# Function to generate the save model path for both custom model and segmentation model.
def generate_save_model_path(PREFIX, model_name, dataset_name, sigma, size, pretrained=None, backbone=None):
    if pretrained is not None:
        pretrained_dir = pretrained
    else:
        pretrained_dir = "no_pretrain"

    if backbone is not None:
        backbone_dir = backbone
    else:
        backbone_dir = "no_backbone"

    save_model_path = f'{PREFIX}/results/models/{model_name}/{pretrained_dir}/{backbone_dir}/{dataset_name}/sigma{sigma}_size{str(size).replace(", ", "x")}'
    return save_model_path

# Generate path if it does not exist
def generate_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# Save the: original image with the heatmaps overlayed, single heatmaps and fused heatmap
def save_heatmaps(batch_images, batch_heatmaps, images_name, save_dir):
    for i, sample in enumerate(batch_heatmaps):
        
        original_image = batch_images[i].permute(1, 2, 0).cpu().numpy()
        
        plt.figure(figsize=(10, 10))
        plt.imshow(original_image)
        for j, heatmap in enumerate(sample):
            plt.imshow(heatmap, cmap='viridis', alpha=0.25)
            
            # Save the single heatmaps
            plt.imsave(f"{save_dir}/{images_name[i]}_heatmap_{j}.png", heatmap, cmap='viridis')

        # Save the original image with the heatmaps overlayed
        plt.savefig(f"{save_dir}/{images_name[i]}_overlayed_heatmaps.png")
        plt.close()
        
        # Save the fused heatmap
        fused_heatmap = fuse_heatmaps(sample)
        plt.imsave(f"{save_dir}/{images_name[i]}_fused_heatmap.png", fused_heatmap, cmap='viridis')
        


# -----------------------------------------------------------------------------------------------------------------##

# Load env variables from a .env file
def load_env_variables(env_file):
    with open(env_file, 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            # trim whitespace
            key = key.strip()
            value = value.strip()
            # remove quotes if present
            if value[0] == value[-1] and value.startswith(("'", '"')):
                value = value[1:-1]

            os.environ[key] = value

def send_telegram_message(text, env_file='~/.env'):
    env_file_path = os.path.expanduser(env_file)

    if os.path.exists(env_file_path):
        load_env_variables(env_file_path)
    else:
        print(f"Error: {env_file_path} does not exist")
        return
    
    token = os.getenv('TELEGRAM_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    url_req = "https://api.telegram.org/bot" + token + "/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': text,
        'parse_mode': 'HTML' #or HTML or MarkdownV2
    }
    
    try:
        response = requests.get(url_req, params=payload)
    except Exception as e:
        print(f"Error sending telegram message: {e}")

    return response
        
        