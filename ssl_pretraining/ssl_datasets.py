


# ------------------------------------------------------------------------
#                               Libraries
# ------------------------------------------------------------------------

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
import albumentations.pytorch



# ------------------------------------------------------------------------
#                               Chest Dataset
# ------------------------------------------------------------------------

# Load the dataset from the train and test folders in the root directory
class ChestDataset(Dataset):
    def __init__(self, root_dir, channels=1, transform=None, phase='train'):

        self.root_dir = root_dir
        self.transform = transform
        self.channels = channels
       
        self.pth_Image = os.path.join(root_dir, 'pngs')
            
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
        if phase == 'train':
            self.image_files = files[:train_num+val_num]
        elif phase == 'test':
            self.image_files = files[-test_num:]
        elif phase == 'all':
            self.image_files = files
        else:
            raise Exception("Unknown phase: {phase}".format(phase=phase))
        

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]

        image = self.read_image(os.path.join(self.pth_Image, image_name + '.png'))

        data_dict = {'name': image_name, 'image': image}   

        return data_dict

    def read_image(self, image_path):

        if self.channels == 3:
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image).astype(np.float32)

        elif self.channels == 1:
            image = Image.open(image_path).convert('L')
            image_np = np.array(image).astype(np.float32)
            image_np = np.expand_dims(image_np, axis=2) # add channel dimension
        else:
            raise ValueError('Channels must be either 1 or 3')
        
        if self.transform:
            image = self.transform(image=image_np)['image']

        return image
        
    
# ------------------------------------------------------------------------
#                               HAND Dataset
# ------------------------------------------------------------------------

# Load the dataset from the train and test folders in the root directory
class HandDataset(Dataset):
    def __init__(self, root_dir, channels=1, transform=None, phase='train'):

        self.root_dir = root_dir
        self.transform = transform
        self.channels = channels
       
        self.pth_Image = os.path.join(root_dir, 'jpg')
            
        # file index
        files = [i[:-4] for i in sorted(os.listdir(self.pth_Image))]

        n = len(files)
        train_num = 550
        val_num = 59
        test_num = n - train_num - val_num
        if phase == 'train':
            self.image_files = files[:train_num+val_num]
        elif phase == 'test':
            self.image_files = files[-test_num:]
        elif phase == 'all':
            self.image_files = files
        else:
            raise Exception("Unknown phase: {phase}".format(phase=phase))
        

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]

        image = self.read_image(os.path.join(self.pth_Image, image_name + '.jpg'))

        data_dict = {'name': image_name, 'image': image}   

        return data_dict

    def read_image(self, image_path):

        if self.channels == 3:
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image).astype(np.float32)

        elif self.channels == 1:
            image = Image.open(image_path).convert('L')
            image_np = np.array(image).astype(np.float32)
            image_np = np.expand_dims(image_np, axis=2)
        else:
            raise ValueError('Channels must be either 1 or 3')
        
        if self.transform:
            image = self.transform(image=image_np)['image']
            
        return image


# ------------------------------------------------------------------------
#                               CEPH Dataset
# ------------------------------------------------------------------------

class CephaloDataset(Dataset):
    def __init__(self, root_dir, channels=1, transform=None, phase='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.channels = channels
       
        self.pth_Image = os.path.join(root_dir, 'jpg')
            
        # file index
        files = [i[:-4] for i in sorted(os.listdir(self.pth_Image))]

        n = len(files)
        train_num = 130
        val_num = 20
        test_num = n - train_num - val_num
        
        if phase == 'train':
            self.image_files = files[:train_num+val_num]
        elif phase == 'test':
            self.image_files = files[-test_num:]
        elif phase == 'all':
            self.image_files = files
        else:
            raise Exception("Unknown phase: {phase}".format(phase=phase))

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_name = self.image_files[idx]

        image = self.read_image(os.path.join(self.pth_Image, image_name + '.jpg'))

        data_dict = {'name': image_name, 'image': image}   

        return data_dict
    
    def read_image(self, image_path):
            
            if self.channels == 3:
                image = Image.open(image_path).convert('RGB')
                image_np = np.array(image).astype(np.float32)
    
            elif self.channels == 1:
                image = Image.open(image_path).convert('L')
                image_np = np.array(image).astype(np.float32)
                image_np = np.expand_dims(image_np, axis=2)
            else:
                raise ValueError('Channels must be either 1 or 3')
            
            if self.transform:
                image = self.transform(image=image_np)['image']
                
            return image
        
        