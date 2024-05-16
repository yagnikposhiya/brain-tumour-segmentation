"""
author: Yagnik Poshiya
github: @yagnikposhiya
organization: NEUROWORK Research Labs
"""

import os
import glob
import numpy as np
import nibabel as nib

from utils.utils import Z_Score_Normalization
def prepareDataset(trainset_path:str) -> str:
    """
    This function is used to prepare dataset and save the dataset into .npy format

    Parameters:
    - trainset_path (str): training directory path

    Returns:
    - (str): path of directory where dataset is stored in .npy format
    """

    # Here t1 type images are not included because those images do not contain rich information related to the brain tumour.

    flair_list = sorted(glob.glob(f'{trainset_path}/structured/*/*flair.nii')) # prepare list of flair images
    t1ce_list = sorted(glob.glob(f'{trainset_path}/structured/*/*t1ce.nii')) # prepare list of t1ce images
    t2_list = sorted(glob.glob(f'{trainset_path}/structured/*/*t2.nii')) # prepare list of t2 images
    mask_list = sorted(glob.glob(f'{trainset_path}/structured/*/*seg.nii'))

    print("- Total flair images: {}".format(len(flair_list))) # total number of flair images
    print("- Total t1ce images: {}".format(len(t1ce_list))) # total number of t1ce images
    print("- Total t2 images: {}".format(len(t2_list))) # total number of t2 images
    print("- Total mask images: {}".format(len(mask_list))) # total number of mask images

    for image_index in range(len(flair_list)):
        flair_image = nib.load(flair_list[image_index]).get_fdata() # load flair image
        flair_image = Z_Score_Normalization(flair_image) # apply z-score normalization

        t1ce_image = nib.load(t1ce_list[image_index]).get_fdata() # load t1ce image
        t1ce_image = Z_Score_Normalization(t1ce_image) # apply z-score normalization

        t2_image = nib.load(t2_list[image_index]).get_fdata() # load t2 image
        t2_image = Z_Score_Normalization(t2_image) # apply z-score normalization

        mask_image = nib.load(mask_list[image_index]).get_fdata() # load mask image
        mask_image = mask_image.astype(np.uint8) # change the datatype
        mask_image[mask_image==4] = 3 #reassign mask value 4 to 3

        stack_image = np.stack([flair_image,t1ce_image,t2_image], axis=3) # stack flair, t1ce, and t2 images

        crop_image = stack_image[56:184, 56:184, 13:141] # crop stack image to (128,128,128,3); remove portion and slices of an image that do not contains any information
        crop_mask = mask_image[56:184, 56:184, 13:141] # crop mask image; remove portion of a mask image that does not contain information about tumour

        val, counts = np.unique(crop_mask, return_counts=True) # calculate unique value and return counts

        if (1-(counts[0]/counts.sum())) > 0.01: # atleast 1% useful volume with labels that are not 0
            crop_mask = np.eye(4)[crop_mask] # perform one-hot encoding on mask image

            if not os.path.exists(f'{trainset_path}/processed'): # check if processed directory exists or not
                os.makedirs(f'{trainset_path}/processed') # if not then create it

                if not os.path.exists(f'{trainset_path}/processed/images'): # check if images directory exists
                    os.makedirs(f'{trainset_path}/processed/images') # if not then create it
                
                if not os.path.exists(f'{trainset_path}/processed/masks'): # check if masks directory exists
                    os.makedirs(f'{trainset_path}/processed/masks') # if not then create it

            np.save(f'{trainset_path}/processed/images/image_{str(image_index)}.npy', crop_image) # save crop image to specified directory
            np.save(f'{trainset_path}/processed/masks/mask_{str(image_index)}.npy', crop_mask) # save crop mask to specified directory
        else:
            print("Image and Mask both are ignored.")

    return f'{trainset_path}processed' # path of directory where dataset is stored in .npy format

