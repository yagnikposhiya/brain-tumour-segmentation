"""
author: Yagnik Poshiya
github: @yagnikposhiya
organization: NEUROWORK Research Labs
"""
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' or any other backend that supports interactive display
# by default backend: FigureCanvasAgg

import os
import numpy as np
import nilearn as nl
import nibabel as nib
import matplotlib.pyplot as plt
import nilearn.plotting as nlplt

from scipy.stats import norm
from skimage.util import montage
from skimage.transform import rotate

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

    print("from showAllTypesOfImages() function:")
    print("- Shape of a flair image: {}".format(image_flair.shape))  # shape of the flair image
    print("- Shape of a T1 image: {}".format(image_t1.shape)) # shape of the T1 image
    print("- Shape of a T1CE image: {}".format(image_t1ce.shape)) # shape of the T1CE image
    print("- Shape of a T2 image: {}".format(image_t2.shape)) # shape of the T2 image
    print("- Shape of a Mask image: {}".format(image_mask.shape)) # shape of the Mask image

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(20,10)) # create figure contains 1 row 5 columns with 20*10 size of all figures.

    SLICE_W = 25 # unknown variable

    ax1.imshow(image_flair[:,:,image_flair.shape[0]//2-SLICE_W]) # visualize flair image
    ax1.set_title('Flair Image') # set title of a figure

    ax2.imshow(image_t1[:,:,image_t1.shape[0]//2-SLICE_W]) # visualize t1 image
    ax2.set_title('T1 Image') # set title of a figure

    ax3.imshow(image_t1ce[:,:,image_t1ce.shape[0]//2-SLICE_W]) # visualize t1ce image
    ax3.set_title('T1CE Image') # set title of a figure

    ax4.imshow(image_t2[:,:,image_t2.shape[0]//2-SLICE_W]) # visualize t2 image
    ax4.set_title('T2 Image') # set title of a figure

    ax5.imshow(image_mask[:,:,image_mask.shape[0]//2-SLICE_W]) # visualize mask image
    ax5.set_title('Mask Image') # set title of a figure

    plt.show() # display the plots

def createMontage(trainset_path:str, image_name:str) -> None:
    """
    This function is used to create montage of a nifti image.

    Parameters:
    - trainset_path (str): training directory path
    - image_name (str): name of the image exists in training directory

    Returns:
    - (None)
    """

    any_image = nib.load(os.path.join(trainset_path,image_name)).get_fdata()
    print("from createMontage() function:")
    print("- Shape of a image: {}".format(any_image.shape))  # shape of the image

    fig, ax1= plt.subplots(1, 1, figsize=(15,15)) # create a figure contains 1 row 1 column with 15*15 size of all figures
    ax1.imshow(rotate(montage(any_image[:,:,:]), 90, resize=True)) # create montage with 90 degree rotated images
    ax1.set_title(image_name) # set title of a figure

    plt.show() # display the plot

def plotAnatomicalImage(trainset_path:str,image_name:str) -> None:
    """
    This function is used to create anatomical plot of an input image.

    Parameters:
    - trainset_path (str): training directory path
    - image_name (str): name of the image exists in training directory

    Returns:
    - (None)
    """

    image = nl.image.load_img(os.path.join(trainset_path,image_name)) # load only image which belongs to given set: {flair, t1, t1ce, t2}

    fig, ax = plt.subplots(nrows=1,figsize=(30,40)) # create figure with 1 row and 30*40 figure size
    nlplt.plot_anat(image, title=f'Anatomical Plot: {image_name}',axes=ax)
    plt.show() # display the plot

def plotEchoPlanarImage(trainset_path:str,image_name:str) -> None:
    """
    This function is used to create echo-planar plot of an input image.

    Parameters:
    - trainset_path (str): training directory path
    - image_name (str): name of the image exists in training directory

    Returns:
    - (None)
    """

    image = nl.image.load_img(os.path.join(trainset_path,image_name)) # load only image which belongs to given set: {flair, t1, t1ce, t2}

    fig, ax = plt.subplots(nrows=1,figsize=(30,40)) # create figure with 1 row and 30*40 figure size
    nlplt.plot_epi(image, title=f'Echo-Planar Plot: {image_name}',axes=ax)
    plt.show() # display the plot

def plotNormalImage(trainset_path:str,image_name:str) -> None:
    """
    This function is used to create normal plot of an input image.

    Parameters:
    - trainset_path (str): training directory path
    - image_name (str): name of the image exists in training directory

    Returns:
    - (None)
    """

    image = nl.image.load_img(os.path.join(trainset_path,image_name)) # load only image which belongs to given set: {flair, t1, t1ce, t2}

    fig, ax = plt.subplots(nrows=1,figsize=(30,40)) # create figure with 1 row and 30*40 figure size
    nlplt.plot_img(image, title=f'Normal Plot: {image_name}',axes=ax)
    plt.show() # display the plot

def plotImageWithROI(trainset_path:str,image_name:str, mask_name:str) -> None:
    """
    This function is used to create normal plot of an input image.

    Parameters:
    - trainset_path (str): training directory path
    - image_name (str): name of the image exists in training directory
    - mask_name (str): name of the image (mask/seg) exists in training directory

    Returns:
    - (None)
    """

    image = nl.image.load_img(os.path.join(trainset_path,image_name)) # load only image which belongs to given set: {flair, t1, t1ce, t2}
    mask = nl.image.load_img(os.path.join(trainset_path,mask_name)) # load the only mask/seg image

    fig, ax = plt.subplots(nrows=1,figsize=(30,40)) # create figure with 1 row and 30*40 figure size
    nlplt.plot_roi(mask, bg_img=image, title=f'Image with Region of Interest: {image_name}',axes=ax)
    plt.show() # display the plot

def Z_Score_Normalization(trainset_path:str,image_name:str) -> None:
    """
    This function is used to calculate z-score normalization of an input image.

    Parameters:
    - trainset_path (str): training directory path
    - image_name (str): name of the image exists in training directory

    Returns:
    - (None)
    """

    image_data = nib.load(os.path.join(trainset_path,image_name)).get_fdata() # load an input image
    selected_image_data = image_data[:,:,95] # select specific channel i.e. 95th channel out of 155 channels

    voxel_values = selected_image_data.ravel() # reshape data to 1D array
    mean = np.mean(voxel_values) # calculate mean
    std_deviation = np.std(voxel_values) # calculate standard deviation

    normalized_data = (selected_image_data - mean)/std_deviation # apply z-score normalization

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10)) # create figure contains 1 row 2 columns with 20*10 figure size

    ax1.imshow(selected_image_data,cmap='gray') # visualize an input image before applying z-score normalization
    ax1.set_title(f'Before applying z-score normalization \n {image_name}') # set figure title
    ax2.imshow(normalized_data, cmap='gray') # visualize an input image after applying z-score normalization
    ax2.set_title(f'After applying z-score normalization \n {image_name}') # set figure title
    plt.show() # display the plots

    # PLOTING HISTOGRAM FOR CHECKING DISTRIBUTION OF DATA BEFORE Z-SCORE NOMALIZATION
    """
    EXPLANATION:
    This plot provides a visual comparison b/w the histogram of voxel intensity values and the PDF (Probability Density Function)
    of the fitted normal distribution, helping to assess how well the voxel intensity values follow a normal distribution. If the 
    PDF curve closely matches the shape of the histogram, it suggests that the voxel intensity values are approximately normaly
    distributed.
    """
    plt.hist(voxel_values, bins=50, density=True, alpha=0.5, color='b', label='Histogram') # plot the histogram of voxel intensity values
    mu, std = norm.fit(voxel_values)
    xmin, xmax = plt.xlim() # retrieve min and max values of x-axis
    x = np.linspace(xmin, xmax, 57600) # generate 57600 (240*240) values b/w min and max values 
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2) # plot the line with color black and width 2

    plt.xlabel("Voxel Intensity") # set xlabel
    plt.ylabel("Probability Density") # set ylabel
    plt.title(f"Normal Distribution of Voxel Intensity Values Before Z-score Normalization \n {image_name}") # set figure title
    plt.show() # display the plot

    # PLOTING HISTOGRAM FOR CHECKING DISTRIBUTION OF DATA AFTER Z-SCORE NOMALIZATION
    selected_normalized_data = normalized_data.ravel() # select 95th slice and reshape data to 1D array
    plt.hist(selected_normalized_data, bins=50, density=True, alpha=0.5, color='b', label='Histogram') # plot the histogram of normalized intensity values
    mu, std = norm.fit(voxel_values)
    xmin, xmax = plt.xlim() # retrieve min and max values of x-axis
    x = np.linspace(xmin, xmax, 57600) # generate 57600 (240*240) values b/w min and max values 
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2) # plot the line with color black and width 2

    plt.xlabel("Voxel Intensity") # set xlabel
    plt.ylabel("Probability Density") # set ylabel
    plt.title(f"Normal Distribution of Voxel Intensity Values After Z-score Normalization \n {image_name}") # set figure title
    plt.show() # display the plot