
import numpy as np
import scipy.spatial.distance as dist
import utilities
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

## -----------------------------------------------------------------------------------------------------------------##
##                                                  Mean Squared Error                                             ##
## -----------------------------------------------------------------------------------------------------------------##

def compute_mse(gt_keypoints, pred_keypoints):
    assert gt_keypoints.shape == pred_keypoints.shape, "The ground truth list has not the same shape of the predicted list"

    # Compute squared differences
    squared_diff = np.square(gt_keypoints - pred_keypoints)

    # Compute mean
    mse = np.mean(squared_diff)

    return mse

## -----------------------------------------------------------------------------------------------------------------##
##                                                  mAp with OKS for heatmaps                                       ##
## -----------------------------------------------------------------------------------------------------------------##
def compute_oks_heatmaps(ground_truth, prediction, sigma):
    distance = dist.cdist(ground_truth, prediction, 'euclidean')
    scale = 1 
    oks = np.exp(-1 * (distance ** 2) / (2 * (sigma**2) * (scale ** 2)))
    return oks

def compute_map_heatmaps(ground_truth_heatmaps, predicted_heatmaps, sigma=0.1, thresholds=np.arange(0.5, 1.0, 0.05)):
    aps = []

    assert ground_truth_heatmaps.shape == predicted_heatmaps.shape, "Heatmaps should have the same shape"

    oks = compute_oks_heatmaps(ground_truth_heatmaps, predicted_heatmaps, sigma)
    for threshold in thresholds:
        tp = np.sum(oks >= threshold)
        fp = np.sum(oks < threshold)
        precision = tp / (tp + fp)
        aps.append(precision)
    map_value = np.mean(aps)

    return map_value


## -----------------------------------------------------------------------------------------------------------------##
##                                                  mAp with OKS for keypoints                                       ##
## -----------------------------------------------------------------------------------------------------------------##
def compute_oks_keypoints(ground_truth, prediction, sigma):
    # Calculate the distance between the ground truth and prediction points
    distance = np.sqrt(np.sum((ground_truth - prediction)**2, axis=1))

    # Calculate the scale, assuming the points are normalized
    #scale = np.max(ground_truth) - np.min(ground_truth)
    scale = 1
    # Calculate the OKS value
    oks = np.exp(-1 * (distance ** 2) / (2 * (sigma**2) * (scale ** 2)))
    return oks


def compute_map_keypoints(ground_truth_keypoints, predicted_keypoints, sigma=0.1, thresholds=np.arange(0.5, 1.0, 0.05)):
    aps = []

    # Calculate OKS value
    oks = compute_oks_keypoints(ground_truth_keypoints, predicted_keypoints, sigma)
    for threshold in thresholds:
        # Calculate precision
        tp = np.sum(oks >= threshold)
        fp = np.sum(oks < threshold)
        precision = tp / (tp + fp)
        aps.append(precision)
    # Calculate the mean average precision
    map_value = np.mean(aps)
    return map_value



## -----------------------------------------------------------------------------------------------------------------##
##                                                  Intersection Over Union                                       ##
## -----------------------------------------------------------------------------------------------------------------##

def compute_iou_heatmaps(heatmap1, heatmap2):

    assert heatmap1.shape == heatmap2.shape, "Heatmaps should have the same shape"

    overlap = np.logical_and(heatmap1, heatmap2)
    union = np.logical_or(heatmap1, heatmap2)
    overlap_area = np.sum(overlap)
    union_area = np.sum(union)
    IoU = overlap_area / union_area
    return IoU



## -----------------------------------------------------------------------------------------------------------------##
##                                                 Aux functions                                       ##
## -----------------------------------------------------------------------------------------------------------------##

from collections.abc import Iterable

def radial(pt1, pt2, factor=1):
    if  not isinstance(factor,Iterable):
        factor = [factor]*len(pt1)
    return sum(((i-j)*s)**2 for i, j,s  in zip(pt1, pt2, factor))**0.5

def cal_all_distance(points, gt_points, factor=1):
    '''
    points: [(x,y,z...)]
    gt_points: [(x,y,z...)]
    return : [d1,d2, ...]
    '''
    n1 = len(points)
    n2 = len(gt_points)
    if n1 == 0:
        print("[Warning]: Empty input for calculating mean and std")
        return 0, 0
    if n1 != n2:
        raise Exception("Error: lengthes dismatch, {}<>{}".format(n1, n2))
    return [radial(p, q, factor) for p, q in zip(points, gt_points)]


## -----------------------------------------------------------------------------------------------------------------##
##                                                  Mean Radial Error (MRE)                                       ##
## -----------------------------------------------------------------------------------------------------------------##

"""
MRE (Mean Radial Error): 
This measures the average euclidean distance between predicted landmarks and ground truth landmarks. 
It is calculated by taking the mean of the list of distances (cal_all_distance).
"""

def compute_mre(distance_list):
    return np.mean(distance_list)

## -----------------------------------------------------------------------------------------------------------------##
##                                                  Successful Detection Rate (SDR)                                       ##
## -----------------------------------------------------------------------------------------------------------------##
"""
SDR (Successful Detection Rate): 
This measures the percentage of predicted landmarks that are within a threshold distance of the ground truth. 
It is calculated by get_sdr which counts the number of distances below each threshold and divides by the total number of landmarks.
"""

def compute_sdr(distance_list, threshold=[2, 2.5, 3, 4, 6, 9, 10]):
    """
    Compute Successful Detection Rate (SDR) in pixel for a given list of distances and thresholds.
    The SDR is the proportion of predicted points that fall within a certain distance threshold from the ground truth points.
    """
    sdr = {}
    n = len(distance_list)

    for th in threshold:
        sdr[th] = sum(d <= th for d in distance_list) / n
    return sdr

## -----------------------------------------------------------------------------------------------------------------##
##                                                  COMPUTE BATCH METRICS                                       ##
## -----------------------------------------------------------------------------------------------------------------##


def compute_batch_metrics(gt_batch_keypoints, gt_batch_heatmaps, pred_batch, image_size, num_landmarks, useHeatmaps, sigma):

    batch_size = pred_batch.shape[0]
    mse_list = []
    map_list1 = []
    map_list2 = []
    iou_list = []
    distance_list = []  

    #sigma = sigma/10
    sigma = 5
        
    # Loop through the batch
    for i in range(batch_size):
        single_gt_keypoints = gt_batch_keypoints[i, :, :].numpy()
        single_gt_heatmaps = gt_batch_heatmaps[i, :, :].numpy()
        single_prediction = pred_batch[i, :, :].numpy()
        single_image_size = tuple(image_size[i].int().tolist())            

        # So when i compare predicted extracted keypoints with original keypoints they all have the same system reference and same origin size
        single_gt_keypoints = utilities.extract_landmarks(single_gt_heatmaps, num_landmarks)

        # Fuse the original heatmaps
        single_gt_heatmaps_fused = utilities.points_to_heatmap(single_gt_keypoints, img_size=single_image_size, sigma=sigma, fuse=True)
        #single_gt_heatmaps_fused = utilities.fuse_heatmaps(single_gt_heatmaps)

        if useHeatmaps:
            # Extract landmarks from the model's output
            single_pred_keypoints = utilities.extract_landmarks(single_prediction, num_landmarks)

            # Upscaling the prediction to the original image size to compute metrics
            single_pred_heatmaps = utilities.points_to_heatmap(single_pred_keypoints, img_size=single_image_size, sigma=sigma, fuse=True)
        else:
            single_pred_keypoints = single_prediction
            single_pred_heatmaps = utilities.points_to_heatmap(single_pred_keypoints, img_size=single_image_size, sigma=sigma, fuse=True)

        gt_scaled_points = np.array(utilities.scale_points(single_gt_keypoints, single_image_size))
        pred_scaled_points = np.array(utilities.scale_points(single_pred_keypoints, single_image_size))

        # Compute Distance list for MRE and SDR
        if num_landmarks == 6:
            physical_factor = 1 # chest
        elif num_landmarks == 19:
            physical_factor = np.array([2400/single_image_size[0], 1935/single_image_size[1]]) * 0.1 # head
            #physical_factor = 0.46875 # ceph
            #physical_factor = 0.9375
        elif num_landmarks == 37:
            physical_factor = 50/radial(gt_scaled_points[0], gt_scaled_points[4]) # hand
        else:
            raise Exception("Error: Unknown number of landmarks")

        cur_distance_list = cal_all_distance(pred_scaled_points, gt_scaled_points, physical_factor)
        distance_list += cur_distance_list

        # Compute MSE
        mse = compute_mse(gt_scaled_points, pred_scaled_points)
        mse_list.append(mse)

        # Compute mAP with keypoints
        map2 = compute_map_keypoints(gt_scaled_points, pred_scaled_points)
        map_list2.append(map2)

        # Compute mAP with heatmaps
        map1 = compute_map_heatmaps(single_gt_heatmaps_fused, single_pred_heatmaps)
        map_list1.append(map1)

        # Compute IoU
        iou = compute_iou_heatmaps(single_gt_heatmaps_fused, single_pred_heatmaps)
        iou_list.append(iou)
        
    return mse_list, map_list1, map_list2, iou_list, distance_list
