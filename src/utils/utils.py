"""
author: Yagnik Poshiya
github: @yagnikposhiya
organization: NEUROWORK Research Labs
"""
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' or any other backend that supports interactive display
# by default backend: FigureCanvasAgg

import os
import torch
import numpy as np
import nilearn as nl
import nibabel as nib
import matplotlib.pyplot as plt
import nilearn.plotting as nlplt

from scipy.stats import norm
from skimage.util import montage
from skimage.transform import rotate
from matplotlib.colors import ListedColormap

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
    print("-- Shape of a flair image: {}".format(image_flair.shape))  # shape of the flair image
    print("-- Shape of a T1 image: {}".format(image_t1.shape)) # shape of the T1 image
    print("-- Shape of a T1CE image: {}".format(image_t1ce.shape)) # shape of the T1CE image
    print("-- Shape of a T2 image: {}".format(image_t2.shape)) # shape of the T2 image
    print("-- Shape of a Mask image: {}".format(image_mask.shape)) # shape of the Mask image

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
    print("-- Shape of a image: {}".format(any_image.shape))  # shape of the image

    fig, ax1= plt.subplots(1, 1, figsize=(15,15)) # create a figure contains 1 row 1 column with 15*15 size of all figures
    ax1.imshow(rotate(montage(any_image[50:-50,:,:]), 90, resize=True)) # create montage with 90 degree rotated images
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

def Z_Score_Normalization_forSingleSlice(trainset_path:str,image_name:str) -> None:
    """
    This function is used to calculate z-score normalization of an input image.

    Parameters:
    - trainset_path (str): training directory path
    - image_name (str): name of the image exists in training directory

    Returns:
    - (None)
    """

    image_data = nib.load(os.path.join(trainset_path,image_name)).get_fdata() # load an input image
    selected_image_data = image_data[:,:,95] # select specific channel i.e. 95th slice out of 155 slices

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

def flipImageHorizontally(trainset_path:str,image_name:str,mask_name:str) -> None:
    """
    This function is used to apply data augmentation modality i.e. Horizontal Flipping

    Parameters:
    - trainset_path (str): training directory path
    - image_name (str): name of the image exists in training directory
    - mask_name (str): name of the image (mask/seg) exists in training directory

    Returns:
    - (None)
    """

    bg_img = nib.load(os.path.join(trainset_path,image_name)).get_fdata() # load an image of type flair, t1, t1ce, t2 only
    mask_img = nib.load(os.path.join(trainset_path,mask_name)).get_fdata() # load an image of type mask/seg only

    flipped_horizontal_bg_img = np.flip(bg_img, axis=0) # flip alongside the first axis (vertical axis)
    flipped_horizontal_mask_img = np.flip(mask_img, axis=0) # flip alongside the first axis (vertical axis)

    fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(20,20)) # create a figure contains 2 rows and 2 cols with 20*20 each figure size
    
    # PLOT ORIGINAL BACKGROUND IMAGE
    ax1.imshow(bg_img[:,:,95]) # visualize flair, t1, t1ce, t2 images only
    ax1.set_title(f'Image before applying horizontal flip \n {image_name}') # set title for figure

    # PLOT FLIPPED BACKGROUND IMAGE
    ax2.imshow(flipped_horizontal_bg_img[:,:,95]) # visualize flair, t1, t1ce, t2 images only after applying flipping
    ax2.set_title(f'Image after applying horizontal flip \n {image_name}') # set title for figure

    plt.subplots_adjust(hspace=0.5) # add white spce between two rows

    # PLOT ORIGINAL MASK IMAGE
    ax3.imshow(mask_img[:,:,95]) # visualize mask/seg image only
    ax3.set_title(f'Mask/Seg Image before applying horizontal flip \n {image_name}') # set title for figure

    # PLOT FLIPPED MASK IMAGE
    ax4.imshow(flipped_horizontal_mask_img[:,:,95]) # visualize mask/seg image only
    ax4.set_title(f'Mask/Seg Image after applying horizontal flip \n {image_name}') # set title for figure

    plt.show() # display the plots

def flipImageVertically(trainset_path:str,image_name:str,mask_name:str) -> None:
    """
    This function is used to apply data augmentation modality i.e. Vertical Flipping

    Parameters:
    - trainset_path (str): training directory path
    - image_name (str): name of the image exists in training directory
    - mask_name (str): name of the image (mask/seg) exists in training directory

    Returns:
    - (None)
    """

    bg_img = nib.load(os.path.join(trainset_path,image_name)).get_fdata() # load an image of type flair, t1, t1ce, t2 only
    mask_img = nib.load(os.path.join(trainset_path,mask_name)).get_fdata() # load an image of type mask/seg only

    flipped_vertical_bg_img = np.flip(bg_img, axis=1) # flip alongside the second axis (horizontal axis)
    flipped_vertical_mask_img = np.flip(mask_img, axis=1) # flip alongside the second axis (horizontal axis)

    fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(20,20)) # create a figure contains 2 rows and 2 cols with 20*20 each figure size
    
    # PLOT ORIGINAL BACKGROUND IMAGE
    ax1.imshow(bg_img[:,:,95]) # visualize flair, t1, t1ce, t2 images only
    ax1.set_title(f'Image before applying vertical flip \n {image_name}') # set title for figure

    # PLOT FLIPPED BACKGROUND IMAGE
    ax2.imshow(flipped_vertical_bg_img[:,:,95]) # visualize flair, t1, t1ce, t2 images only after applying flipping
    ax2.set_title(f'Image after applying vertical flip \n {image_name}') # set title for figure

    plt.subplots_adjust(hspace=0.5) # add white spce between two rows

    # PLOT ORIGINAL MASK IMAGE
    ax3.imshow(mask_img[:,:,95]) # visualize mask/seg image only
    ax3.set_title(f'Mask/Seg Image before applying vertical flip \n {image_name}') # set title for figure

    # PLOT FLIPPED MASK IMAGE
    ax4.imshow(flipped_vertical_mask_img[:,:,95]) # visualize mask/seg image only after appklying flipping
    ax4.set_title(f'Mask/Seg Image after applying vertical flip \n {image_name}') # set title for figure

    plt.show() # display the plots

def Z_Score_Normalization_forImage(trainset_path:str,image_name:str) -> np.ndarray:
    """
    This function is used to calculate z-score normalization of an input image.

    Parameters:
    - trainset_path (str): training directory path
    - image_name (str): name of the image exists in training directory

    Returns:
    - (np.ndarray): normalized image array of shape (240,240,155)
    """

    image_data = nib.load(os.path.join(trainset_path,image_name)).get_fdata() # load an input image
    mean = np.mean(image_data, axis=(0,1,2)) # calculate mean; axis are 0,1,2 because image array shape is (240,240,155)
    std_deviation = np.std(image_data, axis=(0,1,2)) # calculate standard deviation; axis are 0,1,2 because image array shape is (240,240,155)
    normalized_data = (image_data - mean)/std_deviation # apply z-score normalization
    # print("- Max value before Z-Score Normalization: {}".format(image_data.max())) # maximum value in image array before applying z-score normalization
    # print("- Max value before Z-Score Normalization: {}".format(normalized_data.max())) # maximum value in image array after applying z-score normalization

    return normalized_data # shape (240,240,155)

def Z_Score_Normalization(image:np.ndarray) -> np.ndarray:
    """
    This function is used to calculate z-score normalization of an input image.

    Parameters:
    - image (np.ndarray): n-dimensional numpy array of an image

    Returns:
    - (np.ndarray): normalized image array of shape (240,240,155)
    """

    mean = np.mean(image, axis=(0,1,2)) # calculate mean; axis are 0,1,2 because image array shape is (240,240,155)
    std_deviation = np.std(image, axis=(0,1,2)) # calculate standard deviation; axis are 0,1,2 because image array shape is (240,240,155)
    normalized_data = (image - mean)/std_deviation # apply z-score normalization
    # print("- Max value before Z-Score Normalization: {}".format(image_data.max())) # maximum value in image array before applying z-score normalization
    # print("- Max value before Z-Score Normalization: {}".format(normalized_data.max())) # maximum value in image array after applying z-score normalization

    return normalized_data # shape (240,240,155)

def croppedImagePlot(image:np.ndarray,mask:np.ndarray, trainset_path:str, image_names: list) -> None:
    """
    This function is used to plot flair, t1ce, and t2 images after cropping and before cropping.

    Parameters:
    - image (np.ndarray): stacked image contains flair, t1ce, and t2 type images
    - mask (np.ndarray): mask image contains only mask/seg region of tumour
    - trainset_path (str): training directory path
    - image_names (list): list of names of the images exist in training directory

    Returns:
    - (None)
    """

    flair_image = nib.load(os.path.join(trainset_path,image_names[0])).get_fdata() # load flair image
    t1ce_image = nib.load(os.path.join(trainset_path,image_names[2])).get_fdata() # load t1ce image
    t2_image = nib.load(os.path.join(trainset_path,image_names[3])).get_fdata() # load t2 image
    mask_image = nib.load(os.path.join(trainset_path,image_names[-1])).get_fdata() # load mask image

    fig, ((ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax8)) = plt.subplots(nrows=2, ncols=4, figsize=(20,20)) # create a figure canvas with 2 rows and 4 cols with 20*20 figure size each

    ax1.imshow(flair_image[:,:,95]) # visualize normal flair image
    ax1.set_title("Normal flair image") # set figure title

    ax2.imshow(t1ce_image[:,:,95]) # visualize normal t1ce image
    ax2.set_title("Normal t1ce image") # set figure title

    ax3.imshow(t2_image[:,:,95]) # visualize normal t2 image
    ax3.set_title("Normal t2 image") # set figure title

    ax4.imshow(mask_image[:,:,95]) # visualize normal mask image
    ax4.set_title("Normal mask image") # set figure title

    plt.subplots_adjust(hspace=0.5) # add white space between two rows

    ax5.imshow(image[:,:,82,0]) # visualize cropped flair image
    ax5.set_title("Cropped flair image") # set figure title

    ax6.imshow(image[:,:,82,1]) # visualize cropped t1ce image
    ax6.set_title("Cropped t1ce image") # set figure title

    ax7.imshow(image[:,:,82,2]) # visualize cropped t2 image
    ax7.set_title("Cropped t2 image") # set figure title

    ax8.imshow(mask[:,:,82]) # visualize cropped mask image
    ax8.set_title("Cropped mask image") # set figure title

    plt.show() # display the plots

def get_user_choice(start:int, end:int) -> int:
    """
    This function is used to get user choice in integer within specific range.

    Parameters:
    - start (int): starting point of the range
    - end (int): ending point of the range

    Returns:
    - (int): user choice in integer value
    """

    while True:
        try:
            choice = int(input(f"Please enter a choice between {start} and {end}: ")) # ask users to enter their choice
            
            if start <= choice <= end:
                return choice
            else:
                print(f"Invalid choice. Please enter a number between {start} and {end}") # ask user to enter a choice between specified range again

        except ValueError:
            print(f"Invalid choice. Please enter a number between {start} and {end}") # ask user to enter a choice between specified range again


def available_models() -> None:
    """
    This function is used to provide a list of available models for training on the existing dataset.

    Parameters: 
    - (None)

    Returns:
    - (list,int): returns (available models,user choice) tuple
    """

    models = ['UNet',
              'MobileNetV1UNet',
              'MobileNetV2UNet',
              'MobileNetV3UNet (small)',
              'MobileNetV3UNet (large)',
              'Cascaded MobileNetV3UNet (large)',
              'BoxUNet',
              'MobileNetV3UNet (large) without SE Block'] # list of available models
    
    print("Select any one neural network architecture from the list given below")
    for i in range(len(models)):
        print(f"{i}_________{models[i]}") # print list of available models with integer model number

    return models, get_user_choice(0, len(models)-1) # get user choice

def save_trained_model(model, model_prefix: str, path: str) -> None:
    """
    This function is used to save trained models at specified path.

    Parameters:
    - model (any): model file which contains metadata with parameters
    - model_prefix (str): model file will be saved with this prefix (model name)
    - path (str): path to the directory where model will be saved

    Returns:
    - (None)
    """

    if not os.path.exists(path): # check directory exists or not
        os.makedirs(path) # if not then create it

    model_prefix = model_prefix.replace(" ","_") # replace white space if it available in the model name/prefix
    model_prefix = model_prefix.replace("(","").replace(")","") # remove () braces from model prefix

    while True:
        try:
            model_name_from_user = str(input(f"Enter model name (File will be saved with \"{model_prefix}\" prefix): ")) # ask user to enter model name
            
            if not os.path.exists(os.path.join(path,f"{model_prefix}_{model_name_from_user}.pth")):
                break
            else:
                print("File with the same name is already exist. Please enter unique name.") # ask user to enter a filename again

        except ValueError:
            print("File with the same name is already exist. Please enter unique name.") # ask user to enter a filename again   


    torch.save(model.state_dict(),os.path.join(path,f"{model_prefix}_{model_name_from_user}.pth")) # save the at specified path and specified name
    print(f"Trained model is successfully saved at: \n{os.path.join(path,f"{model_prefix}_{model_name_from_user}.pth")}")

def load_saved_model(model_class,num_classes,learning_rate) -> torch.any:
    """
    This function is used to load saved trained models.
    
    Parameters:
    - model_class (any): model class
    - num_classes (int) : number of segmentation classes
    - learning_rate (float): learning rate

    Returns:
    - (None)
    """

    while True:
        try:
            model_path = str(input("Enter path where model file exists (with filename): ")) # ask user to enter path where model file exists
            print("Loading saved model...")

            if not os.path.exists(model_path): # check whether model file is exist or not
                print(f"The file {model_path} does not exist.")
                return None
            
            elif os.path.exists(model_path) and os.path.isfile(model_path):
                model = model_class(num_classes=num_classes,learning_rate=learning_rate) # initiailze model class
                model.load_state_dict(torch.load(model_path)) # load the saved state dictionary
                # model.eval() # set the model to evaluation mode
                return model
            
            else:
                print(f"Is a directory: {model_path}")
            
        except ValueError:
            print(f"The file {model_path} does not exist.")

def prepareImageForInference() -> torch.Tensor:
    """
    This function is used to load and pre-process the image for inference.

    Parameters:
    - (None)

    Returns:
    - (torch.Tensor): return input image in tensor type with numpy type also
    """

    while True:
        try:
            image_path = str(input("Enter path where image file exists (with filename): "))# ask user to enter image path
            mask_path = str(input("Enter path where mask file exists (with filename): ")) # ask user to enter mask path

            print("Preparing image/mask for inference...")

            if (not os.path.exists(image_path) or (not os.path.exists(mask_path))): # check whether image file is exist or not
                print(f"The file either {image_path} or {mask_path} does not exist.")
                return None
            
            elif (os.path.exists(image_path) and os.path.exists(mask_path)) and (os.path.isfile(image_path) and os.path.isfile(mask_path)):
                image = nib.load(image_path).get_fdata() # load input image
                Z_Score_Normalization(image) # apply z-score normalization to image

                mask = nib.load(mask_path).get_fdata() # load mask 

                image = image[56:184, 56:184, 13:141] # crop image and transform image shape to (128,128,128)
                mask = mask[56:184, 56:184, 13:141] # crop mask and transform mask shape to (128,128,128)

                image = image[:,:,62] # select only single slice from an image
                mask = mask[:,:,62] # select only single slice from mask
                mask = mask.astype(np.uint8) # change the data type
                print(f"- Grountruth mask unique before label changing: {np.unique(mask)}")
                mask [mask == 4] = 3 # reassign mask value 4 to 3
                print(f"- Grountruth mask unique after label changing: {np.unique(mask)}")


                np_image = image # copy image which is in numpy format
                np_mask = mask # copy mask which is in numpy format

                image = np.expand_dims(image,axis=0) # add channel dimension to make the shape (1,128,128)

                print(f"- Shape image before converting into torch.tensor: {image.shape}")
                print(f"- Shape mask before converting into torch.tensor: {np_mask.shape}")

                image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) # convert numpy array into torch tensor and add batch dimension to an image
                mask = torch.tensor(mask,dtype=torch.long) # convert to torch tensor

                print(f"- Shape image after converting into torch.tensor: {image.shape}")

                return image, mask, np_image, np_mask
            
            else:
                print(f"Is a directory: either {image_path} or {mask_path}")

        except ValueError:
            print(f"The file either {image_path} or {mask_path} does not exist.")


def groundTruthVSPredicted_AllClasses(image: np.ndarray, groundtruth_mask: np.ndarray, predicted_mask:np.ndarray, model:str) -> None:
    """
    This function is used to plot normal images with groundtruth and predicted ROIs.

    Parameters:
    - image (np.ndarray): tensor normal image
    - groundtruth_mask (np.ndarray): tensor groundtruth mask image
    - predicted_mask (np.ndarray): tensor predicted mask image
    - model (str): name of the neural network architecture

    Returns:
    - (None)
    """

    # define a custom colormap for the mask
    class_colors = ['black','red','green','yellow'] # colors for classes 0, 1, 2, and 3
    cmap = ListedColormap(class_colors)

    print(f"- Groundtruth mask shape: {groundtruth_mask.shape}")
    print(f"- Groundtruth mask: {groundtruth_mask}")

    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,20)) # create a figure canvas with 1 row and 2 columns with 20*20 figure size each
    ax1.imshow(image, cmap='gray', interpolation=None) # display the normal image with grayscale
    ax1.imshow(groundtruth_mask, cmap=cmap, interpolation=None, alpha=0.3) # overlay the mask with transparency
    ax1.set_title(f"Image with groundtruth ROI, model: {model}")
    
    predicted_mask = predicted_mask.permute(1,2,0) # rearrange the dimension from (1,128,128) to (128,128,1)

    ax2.imshow(image, cmap='gray', interpolation=None) # display the normal image with grayscale
    ax2.imshow(predicted_mask, cmap=cmap, interpolation=None, alpha=0.3) # overlay the mask with transparency
    ax2.set_title(f"Image with predicted ROI, model: {model}")

    plt.show() # display the plots

def available_optimizers() -> None:
    """
    This function is used to provide list of available optimizers for model training.

    Parameters:
    - (None)

    Returns:
    - (list,int): returns (available optimizers,user choice) tuple
    """

    optimizers = ['Adam',
                  'AdamW',
                  'RMSProp',
                  'SGD'] # list of available optimizers
    
    print("Select any one optimizer from the list given below")
    for i in range(len(optimizers)):
        print(f"{i}_________{optimizers[i]}") # print list of available optimizers with integer optimizer number

    return optimizers, get_user_choice(0,len(optimizers)-1) # get user choice
