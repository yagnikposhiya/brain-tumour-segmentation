"""
author: Yagnik Poshiya
github: @yagnikposhiya
organization: NEUROWORK Research Labs
"""

import os
import glob
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
import numpy as np
import nibabel as nib
import pytorch_lightning as pl

from PIL import Image
from typing import Any
from utils.utils import Z_Score_Normalization
from torch.utils.data import Dataset, DataLoader, random_split

def prepareDataset(path:str,path_to_save_processed_data:str ,directory:str) -> str:
    """
    This function is used to prepare dataset and save the dataset into .npy format

    Parameters:
    - path (str): training directory path
    - path_to_save_processed_data (str): path of the directory where processed data will be stored
    - directory (str): specify the folder name whether "train" or "valid"

    Returns:
    - (str): path of directory where dataset is stored in .npy format
    """

    # Here t1 type images are not included because those images do not contain rich information related to the brain tumour.

    flair_list = sorted(glob.glob(f'{path}/structured/{directory}/*/*flair.nii')) # prepare list of flair images
    t1ce_list = sorted(glob.glob(f'{path}/structured/{directory}/*/*t1ce.nii')) # prepare list of t1ce images
    t2_list = sorted(glob.glob(f'{path}/structured/{directory}/*/*t2.nii')) # prepare list of t2 images

    if directory=="train":
        mask_list = sorted(glob.glob(f'{path}/structured/{directory}/*/*seg.nii'))

    print("- Total flair images: {}".format(len(flair_list))) # total number of flair images
    print("- Total t1ce images: {}".format(len(t1ce_list))) # total number of t1ce images
    print("- Total t2 images: {}".format(len(t2_list))) # total number of t2 images

    if directory=="train": # print total number of mask images because only train directory contains mask images
        print("- Total mask images: {}".format(len(mask_list))) # total number of mask images

    for image_index in range(len(flair_list)):
        flair_image = nib.load(flair_list[image_index]).get_fdata() # load flair image
        flair_image = Z_Score_Normalization(flair_image) # apply z-score normalization

        t1ce_image = nib.load(t1ce_list[image_index]).get_fdata() # load t1ce image
        t1ce_image = Z_Score_Normalization(t1ce_image) # apply z-score normalization

        t2_image = nib.load(t2_list[image_index]).get_fdata() # load t2 image
        t2_image = Z_Score_Normalization(t2_image) # apply z-score normalization

        if directory=="train":
            mask_image = nib.load(mask_list[image_index]).get_fdata() # load mask image
            mask_image = mask_image.astype(np.uint8) # change the datatype
            mask_image[mask_image==4] = 3 #reassign mask value 4 to 3

        # stack_image = np.stack([flair_image,t1ce_image,t2_image], axis=3) # stack flair, t1ce, and t2 images
        # crop_image = stack_image[56:184, 56:184, 13:141] # crop stack image to (128,128,128,3); remove portion and slices of an image that do not contains any information
        # start from here
        flair_crop_image = flair_image[56:184, 56:184, 13:141] # crop flair image
        t1ce_crop_image = t1ce_image[56:184, 56:184, 13:141] # crop t1ce image
        t2_crop_image = t2_image[56:184, 56:184, 13:141] # crop t2 image

        if directory=="train": # if and only if directory is "train" directory
            crop_mask = mask_image[56:184, 56:184, 13:141] # crop mask image; remove portion of a mask image that does not contain information about tumour
            val, counts = np.unique(crop_mask, return_counts=True) # calculate unique value and return counts

            if (1-(counts[0]/counts.sum())) > 0.01: # atleast 1% useful volume with labels that are not 0
                # crop_mask = np.eye(4)[crop_mask] # perform one-hot encoding on mask image

                if not os.path.exists(f'{path_to_save_processed_data}/processed'): # check if processed directory exists or not
                    os.makedirs(f'{path_to_save_processed_data}/processed') # if not then create it

                    if not os.path.exists(f'{path_to_save_processed_data}/processed/{directory}'): # check if train/valid directory exists
                        os.makedirs(f'{path_to_save_processed_data}/processed/{directory}') # if not then create it

                        if not os.path.exists(f'{path_to_save_processed_data}/processed/{directory}/images'): # check if images directory exists
                            os.makedirs(f'{path_to_save_processed_data}/processed/{directory}/images') # if not then create it
                        
                        if not os.path.exists(f'{path_to_save_processed_data}/processed/{directory}/masks'): # check if masks directory exists
                            os.makedirs(f'{path_to_save_processed_data}/processed/{directory}/masks') # if not then create it
                
                for j in range(3): # because only 3 types of images are there; those are flair, t1ce, t2
                    for i in range(128): # each image contains 128 slices that's why; 128*3 per single directory images
                        if j==0:
                            image = flair_crop_image[:,:,i] # save single slice of cropped image
                        elif j==1:
                            image = t1ce_crop_image[:,:,i] # save single slice of cropped image
                        elif j==2:
                            image = t2_crop_image[:,:,i] # save single slice of cropped image

                        mask = crop_mask[:,:,i] # assign crop_mask array to mask variable for ease

                        np.save(f'{path_to_save_processed_data}/processed/{directory}/images/image_{image_index+1}_{j+1}_{i+1}.npy',image) # save input image in .npy format
                        np.save(f'{path_to_save_processed_data}/processed/{directory}/masks/mask_{image_index+1}_{j+1}_{i+1}.npy',mask) # save mask image in .npy format

                # np.save(f'{path_to_save_processed_data}/processed/{directory}/images/image_{str(image_index)}.npy', crop_image) # save crop image to specified directory
                # np.save(f'{path_to_save_processed_data}/processed/{directory}/masks/mask_{str(image_index)}.npy', crop_mask) # save crop mask to specified directory
            else:
                print("Image and Mask both are ignored.")
            
        else: # if directory is "validation" directory
            if not os.path.exists(f'{path_to_save_processed_data}/processed'): # check if processed directory exists or not
                os.makedirs(f'{path_to_save_processed_data}/processed') # if not then create it

                if not os.path.exists(f'{path_to_save_processed_data}/processed/{directory}'): # check if train/valid directory exists
                    os.makedirs(f'{path_to_save_processed_data}/processed/{directory}') # if not then create it

                if not os.path.exists(f'{path_to_save_processed_data}/processed/{directory}/images'): # check if images directory exists
                    os.makedirs(f'{path_to_save_processed_data}/processed/{directory}/images') # if not then create it
                        
                if not os.path.exists(f'{path_to_save_processed_data}/processed/{directory}/masks'): # check if masks directory exists
                    os.makedirs(f'{path_to_save_processed_data}/processed/{directory}/masks') # if not then create it

            for j in range(3): # because only 3 types of images are there; those are flair, t1ce, t2
                for i in range(128): # each image contains 128 slices that's why; 128*3 per single directory images
                    if j==0:
                        image = flair_crop_image[:,:,i] # save single slice of cropped image
                    elif j==1:
                        image = t1ce_crop_image[:,:,i] # save single slice of cropped image
                    elif j==2:
                        image = t2_crop_image[:,:,i] # save single slice of cropped image

                    np.save(f'{path_to_save_processed_data}/processed/{directory}/images/image_{image_index+1}_{j+1}_{i+1}.npy',image) # save input image in .npy format

            # np.save(f'{path}/processed/{directory}/images/image_{str(image_index)}.npy', crop_image) # save crop image to specified directory

    return f'{path_to_save_processed_data}processed/{directory}' # path of directory where dataset is stored in .npy format

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=False) -> None:
        self.images_dir = images_dir # set path for directory contains input images in .npy format
        self.masks_dir = masks_dir # set path for directory contains masks in .npy format
        self.transform = transform # set boolen value for transform
        self.image_files = sorted(os.listdir(images_dir)) # generate a list of input images
        self.masks_files = sorted(os.listdir(masks_dir)) # generate list of mask/seg images

    def __len__(self) -> int:
        return len(self.image_files) # return total number of data samples available in the dataset
    
    def __getitem__(self, index) -> Any:
        image_path = os.path.join(self.images_dir, self.image_files[index]) # generate path for a single image
        mask_path = os.path.join(self.masks_dir, self.masks_files[index]) # generate path for a single mask image

        raw_image = np.load(image_path) # load a image available in the .npy format
        raw_mask = np.load(mask_path) # load a mask image available in the .npy format
        """
        There is no need to do anything related to one-hot encoding for mask images. Once you have clarify the number of classes
        at the time of the model implementation then it will work in the segmentation task.
        """
        # mask = np.eye(4)[mask] # perform one-hot encoding on mask image
        # print("- Shape of the image stored into .npy format: {}".format(image.shape))
        # print("- Shape of the mask stored into .npy format: {}".format(mask.shape))

        # tensors with negative strides are not currently supported that's why have to use PIL and then have to apply transformations
        raw_image = Image.fromarray(raw_image).convert("L") # convert to PIL Images for transformations
        raw_mask = Image.fromarray(raw_mask).convert("L") # convert to PIL Images for transformations

        if self.transform:
            random_int = torch.randint(0, 3, (1,)).item() # generate a random integer between 0-2
            if random_int == 0:
                image = raw_image # no augmentation techniques are applied
                mask = raw_mask # no augmentation techniques are applied
            elif random_int == 1:
                image = np.flip(raw_image,axis=0) # flip alongside the first axis (vertical axis); horizontal flipping
                mask = np.flip(raw_mask,axis=0) # flip alongside the first axis (vertical axis); horizontal flipping
            elif random_int == 2:
                image = np.flip(raw_image,axis=1) # flip alongside the first axis (horizontal axis); vertical flipping
                mask = np.flip(raw_mask,axis=1) # flip alongside the first axis (horizontal axis); vertical flipping

        # convert back to numpy arrays
        image = np.array(image)
        mask = np.array(mask)

        # ensure no negative strides
        """
        In the context of arrays and tensors, a stride represents the number of elements to skip in memory 
        to move to the next element along each dimension. Strides are crucial in understanding how multi-dimensional arrays 
        are laid out in memory and how to navigate through them.
        """
        image = image.copy()
        mask = mask.copy()

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) # convert to torch tensors; .unsqueeze(0) used for add channel dimension
        mask = torch.tensor(mask, dtype=torch.long) # convert to torch tensors

        return image, mask # return image and mask in the form of the tuple

class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, train_images_dir, train_masks_dir, val_images_dir, val_masks_dir, batch_size=32, transform=False, val_split=0.2, test_split=0.1) -> None:
        super().__init__()
        self.train_images_dir = train_images_dir # set path for directory contains input images for training
        self.train_masks_dir = train_masks_dir # set path for directory contains mask images for training
        self.val_images_dir = val_images_dir # set path for directory contains input images for validation
        self.val_masks_dir = val_masks_dir # set path for directory contains input mask images for validation
        self.batch_size = batch_size # set batch size
        self.transform = transform # set boolean value for transformation
        self.val_split = val_split # set validation split in float-point value
        self.test_split = test_split # set test split in float-point value

    def setup(self, stage=None) -> None:
        self.train_dataset = SegmentationDataset(self.train_images_dir, self.train_masks_dir, transform=self.transform) # create an instance of SegmentationDataset class and load data samples as needed
        # self.val_dataset = SegmentationDataset(self.val_images_dir, self.val_masks_dir) # create an instance of SegmentationDataset class and load data samples as needed
        val_size = int(len(self.train_dataset) * self.val_split) # calculate validation set size
        test_size = int(len(self.train_dataset) * self.test_split) # calculate test set size
        train_size = len(self.train_dataset) - val_size - test_size # calculate train set size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.train_dataset, [train_size, val_size, test_size]) # split whole dataset into train, validation, and test set

    def train_dataloader(self) -> Any:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True) # return train set
        
    def val_dataloader(self) -> Any:
        return DataLoader(self.val_dataset, batch_size=self.batch_size) # return validation set

    # def test_dataloader(self) -> os.Any:
    #      return DataLoader(self.test_dataset, batch_size=self.batch_size) # return test set
