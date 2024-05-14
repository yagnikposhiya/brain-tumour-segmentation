"""
author: Yagnik Poshiya
github: @yagnikposhiya
organization: NEUROWORK Research Labs
"""

import os
import nibabel as nib
import matplotlib.pyplot as plt

def showAllTypesOfImages(trainset_path:str, image_name:list) -> None:
    """
    This function is used to visualize the all types of fmri images.
    Those types are flair, t1, t1ce, t2, and mask image

    Parameters:
    - trainset_path (str): training directory path
    - image_name (list): list of names of the image exists in training directory

    Returns:
    - (None)
    """

    image_flair = nib.load(os.path.join(trainset_path,image_name[0])).get_fdata() # load flair type image data
    image_t1 = nib.load(os.path.join(trainset_path,image_name[1])).get_fdata() # load t1 type image data
    image_t1ce = nib.load(os.path.join(trainset_path,image_name[2])).get_fdata() # load t1ce type image data
    image_t2 = nib.load(os.path.join(trainset_path,image_name[3])).get_fdata() # load t2 type image data
    image_mask = nib.load(os.path.join(trainset_path,image_name[4])).get_fdata() # load mask type image data

    print("Shape of a flair image: {}".format(image_flair.shape))  # shape of the flair image
    print("Shape of a T1 image: {}".fomat(image_t1.shape)) # shape of the T1 image
    print("Shape of a T1CE image: {}".format(image_t1ce.shape)) # shape of the T1CE image
    print("Shape of a T2 image: {}".format(image_t2.shape)) # shape of the T2 image
    print("Shape of a Mask image: {}".format(image_mask.shape)) # shape of the Mask image

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20,10)) # create figure contains 1 row 5 columns with 20*10 size of all figures.

    SLICE_W = 25 # unknown variable

    ax1.imshow(image_flair[:,:,image_flair.shape[0]//2-SLICE_W], cmap='gray') # visualize flair image
    ax1.set_title('Flair Image') # set title of a figure

    ax2.imshow(image_t1[:,:,image_t1.shape[0]//2-SLICE_W], cmap='gray') # visualize t1 image
    ax2.set_title('T1 Image') # set title of a figure

    ax3.imshow(image_t1ce[:,:,image_t1ce.shape[0]//2-SLICE_W], cmap='gray') # visualize t1ce image
    ax3.set_title('T1CE Image') # set title of a figure

    ax4.imshow(image_t2[:,:,image_t2.shape[0]//2-SLICE_W], cmap='gray') # visualize t2 image
    ax4.set_title('T2 Image') # set title of a figure

    ax5.imshow(image_mask[:,:,image_mask.shape[0]//2-SLICE_W], cmap='gray') # visualize mask image
    ax5.set_title('Mask Image') # set title of a figure
