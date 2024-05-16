"""
author: Yagnik Poshiya
github: @yagnikposhiya
organization: NEUROWORK Research Labs
"""

import os
import numpy as np
import nibabel as nib

from config import Config
from brats2020 import prepareDataset
from utils.utils import showAllTypesOfImages
from gpu_config.check import check_gpu_config
from utils.utils import Z_Score_Normalization_forImage, croppedImagePlot

if __name__=='__main__':

    check_gpu_config() # check whether GPUs are available locally

    config = Config() # create an instance of class Config

    # SINGLE DIRECTORY: IMAGE PREPROCESSING

    print("--- FOR SINGLE DIRECTORY: IMAGE PROCESSING ---")

    print("Visualizing all types of NIFTI images ...")
    showAllTypesOfImages(config.TRAINSET_PATH,config.TRAIN_IMAGE_PATH) # visualize all types of images for specific record
    print("Visualization finished.")

    print("Applying Z-Score Normalization ...")
    normalized_flair_image = Z_Score_Normalization_forImage(config.TRAINSET_PATH,config.TRAIN_IMAGE_PATH[0]) # apply z-score normalization on whole flair image array
    normalized_t1ce_image = Z_Score_Normalization_forImage(config.TRAINSET_PATH,config.TRAIN_IMAGE_PATH[2]) # apply z-score normalization on whole t1ce image array
    normalized_t2_image = Z_Score_Normalization_forImage(config.TRAINSET_PATH,config.TRAIN_IMAGE_PATH[3]) # apply z-score normalization on whole t2 image array
    # t1 type image is not included because it does not contain rich information related to brain tumour
    print("Z-Score Normalization is applied successfully.")

    print("Stacking Flair, T1CE, and T2 type images...")
    stack_image = np.stack([normalized_flair_image,normalized_t1ce_image,normalized_t2_image], axis=3) # create new axis and stack all normalized images
    print("- Shape of stacked image: {}".format(stack_image.shape)) # shape of stacked image; i.e. (240,240,155,3)
    print("Stacked Flair, T1CE, and T2 type images.")

    print("Cropping stacked image...")
    crop_stack_image = stack_image[56:184,56:184,13:141] # crop stack image to remove part of an image which does not contain any information
    print("- Shape of cropped stack image: {}".format(crop_stack_image.shape)) # output shape; (128,128,128,3)
    print("Stacked image is cropped successfully.")

    print("Loading a mask image...")
    mask_image = nib.load(os.path.join(config.TRAINSET_PATH,config.TRAIN_IMAGE_PATH[-1])).get_fdata() # load mask/seg image
    print("- Shape of mask image: {}".format(mask_image.shape)) # (240,240,155)
    print("Mask image is loaded successfully.")
    mask_image = mask_image.astype(np.uint8) # change datatype of mask image to np.uint8
    mask_image[mask_image==4] = 3 # if label 4 exists in the mask image then rename it to 3

    print("Cropping mask image...")
    crop_mask_image = mask_image[56:184,56:184,13:141] # crop mask image to remove part of the image which does not contain information and also remove those slies which does not contain any information
    print("- Shape of cropped mask image: {}".format(crop_mask_image.shape)) # (128,128,128)
    print("- Lables contain by cropped mask image: {}".format(np.unique(crop_mask_image))) # [0,1,2,3] instead of [0,1,2,4]
    print("Mask image cropped successfully.")

    croppedImagePlot(crop_stack_image,crop_mask_image,config.TRAINSET_PATH,config.TRAIN_IMAGE_PATH) # plot flair, t1ce, t2, and mask images before and after applying cropping

    print("Performing One-Hot encoding for mask image...")
    encoded_mask_image = np.eye(4)[crop_mask_image] # perform one-hot encoding on mask image
    print("One-Hot encoding is performed successully.")
    print("- Shape of mask image after performing one-hot encoding: {}".format(encoded_mask_image.shape)) # (128,128,128,4)

    # FOR MULTIPLE DIRECTORIES: IMAGE PROCESSING

    print("--- FOR MULTIPLE DIRECTORIES: IMAGE PROCESSING ---")

    print("Preparing dataset...")
    dir_path = prepareDataset(config.TRAINSET_PATH) # prepare dataset and stored it into .npy format
    print("Dataset is prepared and stored into .npy format at: \n{}".format(dir_path))

