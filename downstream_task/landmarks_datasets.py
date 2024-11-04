import numpy as np
import pandas as pd
from PIL import Image
import os

import torch
import utilities
from skimage.transform import resize

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
## -----------------------------------------------------------------------------------------------------------------##
##                                                          CHEST DATASET                                           ##
## -----------------------------------------------------------------------------------------------------------------##
"""
LINK: https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels

X-ray images in this data set have been acquired from the tuberculosis control program of the Department of Health and Human Services of Montgomery County, MD, USA. 
This set contains 138 posterior-anterior x-rays, of which 80 x-rays are normal and 58 x-rays are abnormal with manifestations of tuberculosis. 
All images are de-identified and available in DICOM format. The set covers a wide range of abnormalities, including effusions and miliary patterns.
"""
class Chest(torch.utils.data.Dataset):

    def __init__(self, prefix, phase, size=(512, 512), num_channels=1, fuse_heatmap=False, sigma=8):
        self.phase = phase
        self.new_size = size
        self.dataset_name = 'Chest'

        self.transforms = self.get_transforms()
        self.num_channels = num_channels
        self.fuse_heatmap = fuse_heatmap
        self.sigma = sigma
        self.num_landmarks = 6
        self.pth_Image = os.path.join(prefix, 'pngs')
        self.pth_Label = os.path.join(prefix, 'labels')

        # file index
        files = [i[:-4] for i in sorted(os.listdir(self.pth_Image))]

        exclude_list = ['CHNCXR_0059_0', 'CHNCXR_0178_0', 'CHNCXR_0228_0', 'CHNCXR_0267_0', 'CHNCXR_0295_0', 'CHNCXR_0310_0', 'CHNCXR_0285_0', 'CHNCXR_0276_0', 'CHNCXR_0303_0']
        if exclude_list is not None:
            st = set(exclude_list)
            files = [f for f in files if f not in st]

        n = len(files)
        train_num = 195  
        val_num = 34 
        test_num = n - train_num - val_num
        if self.phase == 'train':
            self.indexes = files[:train_num]
        elif self.phase == 'validate':
            self.indexes = files[train_num:-test_num]
        elif self.phase == 'test':
            self.indexes = files[-test_num:]
        elif self.phase == 'all':
            self.indexes = files
        else:
            raise Exception("Unknown phase: {phase}".format(phase=phase))

    def __getitem__(self, index):
        name = self.indexes[index]
        ret = {'name': name}

        img, img_size= self.readImage(os.path.join(self.pth_Image, name + '.png'))
        points = self.readLandmark(name)        
        heatmaps = utilities.points_to_heatmap(points, sigma=self.sigma, img_size=self.new_size, fuse=self.fuse_heatmap)

        transformed = self.transforms(image=img, masks=heatmaps)
        
        # img shape: CxHxW | heatmaps is a list of CxHxW: example: [CxHxW, CxHxW, CxHxW, CxHxW, CxHxW, CxHxW]
        img, heatmaps = transformed['image'], transformed['masks']
        
        # Image is a torch tensor [C, H, W]
        ret['image'] = img
        ret['landmarks'] = torch.FloatTensor(points)
        # Convert heatmaps to torch tensor [C, H, W]. Stack to give new dimension and float32 type to avoid error in loss function
        ret['heatmaps'] = torch.stack([hm.float() for hm in heatmaps])
        ret['original_size'] = torch.FloatTensor(img_size)
        ret['resized_size'] = torch.FloatTensor(self.new_size)

        return ret

    def __len__(self):
        return len(self.indexes)

    def readLandmark(self, name):
        path = os.path.join(self.pth_Label, name + '.txt')
        points = []
        with open(path, 'r') as f:
            n = int(f.readline())
            for i in range(n):
                ratios = [float(i) for i in f.readline().split()]
                points.append(ratios)
        return np.array(points)
    
    def readImage(self, path):

        if self.num_channels == 3:
            img = Image.open(path).convert('RGB')
            arr = np.array(img).astype(np.float32)
 
        elif self.num_channels == 1:
            img = Image.open(path).convert('L')
            arr = np.array(img).astype(np.float32)
            arr = np.expand_dims(arr, 2)
        else:
            raise ValueError('Channels must be either 1 or 3')

        # Original size in (width, height)
        origin_size = img.size
        resized_image = resize(arr, (self.new_size[0], self.new_size[1], self.num_channels))

        return resized_image, origin_size

    def get_transforms(self):
        if self.phase == 'train':
            return A.Compose([
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0, rotate_limit=2, border_mode=cv2.BORDER_REPLICATE, p=0.5),
            #A.Perspective(scale=(0, 0.02), pad_mode=cv2.BORDER_REPLICATE, p=0.5),
            #A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.2, 0.2), p=0.5),
            #A.Resize(self.new_size[0], self.new_size[1]),
            A.Normalize(normalization='min_max'),
            A.pytorch.ToTensorV2()
        ])
        elif self.phase == 'validate':
            return A.Compose([
            #A.Resize(self.new_size[0], self.new_size[1]),
            A.Normalize(normalization='min_max'),
            A.pytorch.ToTensorV2()
            ])
        elif self.phase == 'test':
            return A.Compose([
            #A.Resize(self.new_size[0], self.new_size[1]),
            A.Normalize(normalization='min_max'),
            A.pytorch.transforms.ToTensorV2()
            ])
        else:
            raise ValueError('phase must be either "train" or "validate" or "test"')

## -----------------------------------------------------------------------------------------------------------------##
##                                                          HAND DATASET                                            ##
## -----------------------------------------------------------------------------------------------------------------##

"""
LINK: https://ipilab.usc.edu/research/baaweb/
ASI: Asian; BLK: African American; CAU: Caucasian; HIS: Hispanic.
"""

class Hand(torch.utils.data.Dataset):

    def __init__(self, prefix, phase, size=(512, 368), num_channels=1, fuse_heatmap=False, sigma=5):

        self.phase = phase
        self.new_size = size
        self.dataset_name = 'Hand'
        
        self.transforms = self.get_transforms()
        self.num_channels = num_channels
        self.fuse_heatmap = fuse_heatmap
        self.sigma = sigma
        self.num_landmarks = 37

        self.pth_Image = os.path.join(prefix, 'jpg')
        self.labels = pd.read_csv(os.path.join(
            prefix, 'labels/all.csv'), header=None, index_col=0)

        # file index
        index_set = set(self.labels.index) # Set of all the labels
        files = [i[:-4] for i in sorted(os.listdir(self.pth_Image))] # -4 to cut ".jpg" # List of all the images
        files = [i for i in files if int(i) in index_set] # List of filters that has a label

        n = len(files)
        train_num = 550
        val_num = 59
        test_num = n - train_num - val_num

        if phase == 'train':
            self.indexes = files[:train_num]
        elif phase == 'validate':
            self.indexes = files[train_num:-test_num]
        elif phase == 'test':
            self.indexes = files[-test_num:]
        elif phase == 'all':
            self.indexes = files
        else:
            raise Exception("Unknown phase: {phase}".format(phase=phase))

    def __getitem__(self, index):
        name = self.indexes[index]
        ret = {'name': name}

        img, img_size = self.readImage(
            os.path.join(self.pth_Image, name + '.jpg'))

        points = self.readLandmark(name, img_size)
        heatmaps = utilities.points_to_heatmap(points, sigma=self.sigma, img_size=self.new_size, fuse=self.fuse_heatmap)

        transformed = self.transforms(image=img, masks=heatmaps)
        img, heatmaps = transformed['image'], transformed['masks']
        
        ret['image'] = img
        ret['landmarks'] = torch.FloatTensor(points)
        ret['heatmaps'] = torch.stack([hm.float() for hm in heatmaps])
        ret['original_size'] = torch.FloatTensor(img_size)
        ret['resized_size'] = torch.FloatTensor(self.new_size)

        return ret
    

    def __len__(self):
        return len(self.indexes)

    def readLandmark(self, name, origin_size):
        li = list(self.labels.loc[int(name), :])   
        points = []
        for i in range(0, len(li), 2):
            ratios = (li[i] / origin_size[0], li[i + 1] / origin_size[1])
            points.append(ratios)
        return np.array(points)

    def readImage(self, path):
        
        if self.num_channels == 3:
            img = Image.open(path).convert('RGB')
            arr = np.array(img).astype(np.float32)
 
        elif self.num_channels == 1:
            img = Image.open(path).convert('L')
            arr = np.array(img).astype(np.float32)
            arr = np.expand_dims(arr, 2)
        else:
            raise ValueError('Channels must be either 1 or 3')
            
        # Original size in (width, height)
        origin_size = img.size
        resized_image = resize(arr, (self.new_size[0], self.new_size[1], self.num_channels))

        return resized_image, origin_size
    
    def get_transforms(self):
        if self.phase == 'train':
            return A.Compose([
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=(-0.02, 0.02), rotate_limit=2, border_mode=cv2.BORDER_REPLICATE, p=0.5),
            #A.Perspective(scale=(0, 0.02), pad_mode=cv2.BORDER_REPLICATE, p=0.5),
            #A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.2, 0.2), p=0.5),
            #A.Resize(self.new_size[0], self.new_size[1]),
            A.Normalize(normalization='min_max'),
            A.pytorch.ToTensorV2()
        ])
        elif self.phase == 'validate':
            return A.Compose([
            #A.Resize(self.new_size[0], self.new_size[1]),
            A.Normalize(normalization='min_max'),
            A.pytorch.ToTensorV2()
            ])
        elif self.phase == 'test':
            return A.Compose([
            #A.Resize(self.new_size[0], self.new_size[1]),
            A.Normalize(normalization='min_max'),
            A.pytorch.transforms.ToTensorV2()
            ])
        else:
            raise ValueError('phase must be either "train" or "validate" or "test"')



## -----------------------------------------------------------------------------------------------------------------##
##                                                          CEPHALOMETRIC DATASET                                            ##
## -----------------------------------------------------------------------------------------------------------------##

"""
LINK: https://www.kaggle.com/datasets/c34a0ef0cd3cfd5c5afbdb30f8541e887171f19f196b1ad63790ca5b28c0ec93
https://figshare.com/s/37ec464af8e81ae6ebbf?file=5466581
"""


class Cephalo(torch.utils.data.Dataset):

    def __init__(self, prefix, phase, size=(512, 416), num_channels=1, fuse_heatmap=False, sigma=5):
        self.phase = phase
        self.new_size = size
        self.dataset_name = 'Cephalo'
        
        self.transforms = self.get_transforms()
        self.num_channels = num_channels
        self.fuse_heatmap = fuse_heatmap
        self.sigma = sigma
        
        self.num_landmarks = 19


        self.pth_Image = os.path.join(prefix, 'jpg')
        self.pth_label_junior = os.path.join(prefix, '400_junior')
        self.pth_label_senior = os.path.join(prefix, '400_senior')

        # file index
        files = [i[:-4] for i in sorted(os.listdir(self.pth_Image))]
        n = len(files)
        
        if phase == 'train':
            self.indexes = files[:130]
        elif phase == 'validate':
            self.indexes = files[130:150]
        elif phase == 'test':
            self.indexes = files[150:400]
        elif phase == 'all':
            self.indexes = files
        else:
            raise Exception("Unknown phase: {phase}".format(phase=phase))


    def __getitem__(self, index):
        name = self.indexes[index]
        ret = {'name': name}

        img, img_size = self.readImage(os.path.join(self.pth_Image, name+'.jpg'))
        points = self.readLandmark(name, img_size)
        heatmaps = utilities.points_to_heatmap(points, sigma=self.sigma, img_size=self.new_size, fuse=self.fuse_heatmap)

        transformed = self.transforms(image=img, masks=heatmaps)        
        img, heatmaps = transformed['image'], transformed['masks']
        
        ret['image'] = img
        ret['landmarks'] = torch.FloatTensor(points)
        ret['heatmaps'] = torch.stack([hm.float() for hm in heatmaps])
        ret['original_size'] = torch.FloatTensor(img_size)
        ret['resized_size'] = torch.FloatTensor(self.new_size)

        return ret

    def __len__(self):
        return len(self.indexes)

    def readLandmark(self, name, origin_size):
        points = []
        with open(os.path.join(self.pth_label_junior, name + '.txt')) as f1:
            with open(os.path.join(self.pth_label_senior, name + '.txt')) as f2:
                for i in range(self.num_landmarks):
                    landmark1 = f1.readline().rstrip('\n').split(',')
                    landmark2 = f2.readline().rstrip('\n').split(',')
                    # Average of junior and senior landmarks
                    landmark = [(float(i) + float(j)) / 2 for i, j in zip(landmark1, landmark2)]
                    #landmark = [float(i) for i in landmark1] 
                    ratios = (landmark[0] / origin_size[0], landmark[1] / origin_size[1])
                    points.append(ratios)
        return np.array(points)

    def readImage(self, path):

        if self.num_channels == 3:
            img = Image.open(path).convert('RGB')
            arr = np.array(img).astype(np.float32)
 
        elif self.num_channels == 1:
            img = Image.open(path).convert('L')
            arr = np.array(img).astype(np.float32)
            arr = np.expand_dims(arr, 2)
        else:
            raise ValueError('Channels must be either 1 or 3')

        # Original size in (width, height)
        origin_size = img.size
        resized_image = resize(arr, (self.new_size[0], self.new_size[1], self.num_channels))
        
        return resized_image, origin_size
    
    def get_transforms(self):
        if self.phase == 'train':
            return A.Compose([
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=(-0.02, 0.02), rotate_limit=2, border_mode=cv2.BORDER_REPLICATE, p=0.5),
            #A.Perspective(scale=(0, 0.02), pad_mode=cv2.BORDER_REPLICATE, p=0.5),
            #A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.2, 0.2), p=0.5),
            #A.Resize(self.new_size[0], self.new_size[1]),
            A.Normalize(normalization='min_max'),
            A.pytorch.ToTensorV2()
        ])
        elif self.phase == 'validate':
            return A.Compose([
            #A.Resize(self.new_size[0], self.new_size[1]),
            A.Normalize(normalization='min_max'),
            A.pytorch.ToTensorV2()
            ])
        elif self.phase == 'test':
            return A.Compose([
            #A.Resize(self.new_size[0], self.new_size[1]),
            A.Normalize(normalization='min_max'),
            A.pytorch.transforms.ToTensorV2()
            ])
        else:
            raise ValueError('phase must be either "train" or "validate" or "test"')


