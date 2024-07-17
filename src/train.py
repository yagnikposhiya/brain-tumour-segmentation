"""
author: Yagnik Poshiya
github: @yagnikposhiya
organization: Neurowork Research Labs
"""

import os
import wandb
import torch
import numpy as np
import nibabel as nib
import pytorch_lightning as pl

# import torchviz
# import graphviz

from config.config import Config
from torchinfo import summary
from nn_arch.unet import UNet
from nn_arch.boxunet import BoxUNet
from brats2020 import prepareDataset
from brats2020 import SegmentationDataModule
from utils.utils import showAllTypesOfImages
from gpu_config.check import check_gpu_config
from nn_arch.mobilenetv1 import MobileNetV1UNet
from nn_arch.mobilenetv2 import MobileNetV2UNet
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from nn_arch.mobilenetv3_small import MobileNetV3SmallUNet
from nn_arch.mobilenetv3_large import MobileNetV3LargeUNet
from nn_arch.cascaded_mobilenetv3_large import CascadedMobileNetV3LargeUNet
from nn_arch.cascaded_mobilenetv3_large_MHA import CascadedMobileNetV3LargeUNet_WithMHA
from nn_arch.mobilenetv3_large_without_SEblock import MobileNetV3LargeUNet_Without_SEBlock
from nn_arch.mobilenetv3_small_without_SEblock import MobileNetV3SmallUNet_Without_SEBlock
from utils.utils import flipImageHorizontally, flipImageVertically, Z_Score_Normalization_forSingleSlice
from utils.utils import Z_Score_Normalization_forImage, croppedImagePlot, available_models, save_trained_model, available_optimizers

# from torchsummary import summary; old name is torchsummary, new name is torchinfo


if __name__=='__main__':

    check_gpu_config() # check whether GPUs are available locally

    config = Config() # create an instance of class Config

    wandb.init(entity=config.ENTITY,
               project=config.PROJECT,
               anonymous=config.ANONYMOUS,
               reinit=config.REINIT) # initialize the weights and biases cloud server instance
    # not passing "group" parameter in the weights and biases initialization process becuase it is not creating groups automatically.
    
    config.printConfiguration() # print all configuration related to paths and model trainig parameters

    # SINGLE DIRECTORY: IMAGE PREPROCESSING

    print("----------------------------------------------")
    print("--- FOR SINGLE DIRECTORY: IMAGE PROCESSING ---")
    print("----------------------------------------------")

    print("Visualizing all types of NIFTI images ...")
    showAllTypesOfImages(config.TRAINSET_PATH,config.TRAIN_IMAGE_PATH) # visualize all types of images for specific record
    print("Visualization finished.")

    print("Applying Z-Score Normalization ...")
    normalized_flair_image = Z_Score_Normalization_forImage(config.TRAINSET_PATH,config.TRAIN_IMAGE_PATH[0]) # apply z-score normalization on whole flair image array
    normalized_t1_image = Z_Score_Normalization_forImage(config.TRAINSET_PATH,config.TRAIN_IMAGE_PATH[1]) # apply z-score normalization on whole t1 image array
    normalized_t1ce_image = Z_Score_Normalization_forImage(config.TRAINSET_PATH,config.TRAIN_IMAGE_PATH[2]) # apply z-score normalization on whole t1ce image array
    normalized_t2_image = Z_Score_Normalization_forImage(config.TRAINSET_PATH,config.TRAIN_IMAGE_PATH[3]) # apply z-score normalization on whole t2 image array
    # t1 type image is not included because it does not contain rich information related to brain tumour
    print("Z-Score Normalization is applied successfully.")

    print("Stacking Flair, T1CE, and T2 type images...")
    stack_image = np.stack([normalized_flair_image,normalized_t1_image,normalized_t1ce_image,normalized_t2_image], axis=3) # create new axis and stack all normalized images
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

    print("Flipping image horizontally...")
    flipImageHorizontally(config.TRAINSET_PATH,config.TRAIN_IMAGE_PATH[0],config.TRAIN_IMAGE_PATH[-1]) # flip image horizontally
    print("Image is flipped horizontally successfully.")

    print("Flipping image vertically...")
    flipImageVertically(config.TRAINSET_PATH,config.TRAIN_IMAGE_PATH[0],config.TRAIN_IMAGE_PATH[-1]) # flip image horizontally
    print("Image is flipped vertically successfully.")

    print("Applying Z-score normalization...")
    Z_Score_Normalization_forSingleSlice(config.TRAINSET_PATH,config.TRAIN_IMAGE_PATH[1]) # apply z-score normalization and visualize image before/after z-score normalization
    print("Z-score normalization is applied successfully.")

    print("Performing One-Hot encoding for mask image...")
    encoded_mask_image = np.eye(4)[crop_mask_image] # perform one-hot encoding on mask image
    print("One-Hot encoding is performed successully.")
    print("- Shape of mask image after performing one-hot encoding: {}".format(encoded_mask_image.shape)) # (128,128,128,4)

    # FOR MULTIPLE DIRECTORIES: IMAGE PROCESSING

    print("--------------------------------------------------")
    print("--- FOR MULTIPLE DIRECTORIES: IMAGE PROCESSING ---")
    print("--------------------------------------------------")

    print("Preparing dataset...")
    print("Preparing trainset...")
    dir_path = prepareDataset(config.TRAINSET_PATH, config.PATH_TO_SAVE_PROCESSED_DATA,"train") # prepare dataset and stored it into .npy format
    print("Trainset is prepared and stored into .npy format at: \n{}".format(dir_path))

    # Here validation set does not contain any mask images so not transforming it into .npy
    # print("Preparing validation set...")
    # dir_path = prepareDataset(config.VALIDSET_PATH,"valid") # prepare dataset and stored it into .npy format
    # print("Validationset is prepared and stored into .npy format at: \n{}".format(dir_path))
    print("Dataset is prepared.")

    config.TRAIN_IMAGES_DIR = os.path.join(dir_path,'images') # set path for directory where images are stored into .npy format
    config.TRAIN_MASKS_DIR = os.path.join(dir_path,'masks') # set path for directory where mask images are stored into .npy format

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # set device for model training
    data_module = SegmentationDataModule(config.TRAIN_IMAGES_DIR,config.TRAIN_MASKS_DIR,config.VAL_IMAGES_DIR,config.VAL_MASKS_DIR,config.BATCH_SIZE, transform=config.TRANSFORM) # initialize the data module

    print("-------------------------------------------------")
    print("------- NN ARCHITECTURE (MODEL) SELECTION -------")
    print("-------------------------------------------------")

    avail_models, user_choice = available_models() # get user choice for available models for training
    avail_optim, user_choice_optim = available_optimizers() # get user choice for available optimizers for training
    print(f"- You have selected {avail_optim[user_choice_optim]} optimizer with {avail_models[user_choice]} neural network architecture.")

    if user_choice == 0:
        model = UNet(num_classes=config.NUM_CLASSES, learning_rate=config.LEARNING_RATE, optimizer=avail_optim[user_choice_optim]) # create a normal standard unet model
        monitor_loss = 'unet_valid_loss' # set string value same as set in model configuration
    elif user_choice == 1:
        model = MobileNetV1UNet(num_classes=config.NUM_CLASSES,learning_rate=config.LEARNING_RATE, optimizer=avail_optim[user_choice_optim]) # create MobileNetV1 model
        monitor_loss = 'mobilenetv1_valid_loss' # set string value same as set in model configuration
    elif user_choice == 2:
        model = MobileNetV2UNet(num_classes=config.NUM_CLASSES, learning_rate=config.LEARNING_RATE, optimizer=avail_optim[user_choice_optim]) # create MobileNetV2 model
        monitor_loss = 'mobilenetv2_valid_loss' # set string value same as set in model configuration
    elif user_choice == 3:
        model = MobileNetV3SmallUNet(num_classes=config.NUM_CLASSES,learning_rate=config.LEARNING_RATE, optimizer=avail_optim[user_choice_optim]) # create MobileNetV3-Small model
        monitor_loss = 'mobilenetv3_small_valid_loss' # set string value same as set in model configuration
    elif user_choice == 4:
        model = MobileNetV3LargeUNet(num_classes=config.NUM_CLASSES,learning_rate=config.LEARNING_RATE, optimizer=avail_optim[user_choice_optim]) # create MobileNetV3-Large model
        monitor_loss = 'mobilenetv3_large_valid_loss' # set string value same as set in model configuration
    elif user_choice == 5:
        model = CascadedMobileNetV3LargeUNet(num_classes=config.NUM_CLASSES, learning_rate=config.LEARNING_RATE, optimizer=avail_optim[user_choice_optim]) # create Cascaded MobileNetV3-Large model
        monitor_loss = 'cascaded_mobilenetv3_large_valid_loss' # set string value same as set in model configuration
    elif user_choice == 6:
        model = BoxUNet(num_classes=config.NUM_CLASSES, learning_rate=config.LEARNING_RATE, optimizer=avail_optim[user_choice_optim]) # create BoxUNet model
        monitor_loss = 'boxunet_valid_loss' # set string value same as set in model configuration
    elif user_choice == 7:
        model = MobileNetV3LargeUNet_Without_SEBlock(num_classes=config.NUM_CLASSES, learning_rate=config.LEARNING_RATE, optimizer=avail_optim[user_choice_optim]) # create MobileNetV3-Large architecture without SE Block
        monitor_loss = 'mobilenetv3_large_without_SE_valid_loss' # set string value same as set in model configuration
    elif user_choice == 8:
        model = MobileNetV3SmallUNet_Without_SEBlock(num_classes=config.NUM_CLASSES, learning_rate=config.LEARNING_RATE, optimizer=avail_optim[user_choice_optim]) # create MobileNetV3-Small architecture without SE Block
        monitor_loss = 'mobilenetv3_small_without_SE_valid_loss' # set string value same as set in model configuration
    elif user_choice == 9:
        model = CascadedMobileNetV3LargeUNet_WithMHA(num_classes=config.NUM_CLASSES, learning_rate=config.LEARNING_RATE, optimizer=avail_optim[user_choice_optim]) # create Cascaded MobileNetV3-Large with MHA mechanism architecture
        monitor_loss = 'cascaded_mobilenetv3_large_with_MHA_valid_loss' # set string value same as set in model configuration
        
    print("- Model summary:\n")
    summary(model,input_size=(1,4,128,128),col_names=["input_size", "output_size", "kernel_size"]) # print model summary; input shape is extracted @ data loading time

    # dummy_input = torch.randn(1,4,128,128) # create dummy input tensor
    # dot = torchviz.make_dot(model(dummy_input), params=dict(model.named_parameters())) # create object of graph
    # dot.render('unet_model', format='png', view=True) # print graph and save it into .png format

    model = model.to(device) # move model architecture to available computing device
    wandb_logger = WandbLogger(log_model=config.LOG_MODEL)

    # initialize early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor=monitor_loss, # any other metric, is being monitored
        patience=10, # number of epochs with no improvements after which training will be stopped
        mode='min' # 'min' because want to stop when loss stops decreasing
    )
    # trainer = pl.Trainer(max_epochs=config.MAX_EPOCHS, log_every_n_steps=1, logger=wandb_logger, callbacks=early_stopping_callback) # set the maxium number of epochs; saving a training logs at every step; also applied early stopping
    trainer = pl.Trainer(max_epochs=config.MAX_EPOCHS, log_every_n_steps=1, logger=wandb_logger) # set the maxium number of epochs; saving a training logs at every step; without early stopping

    print("-------------------------------------------------")
    print("------- NN ARCHITECTURE (MODEL) TRAINING --------")
    print("-------------------------------------------------")
    print("Training started...")
    trainer.fit(model,data_module) # train the normal standard unet model
    print("Training finished.")

    wandb.finish() # close the weights and biases cloud instance

    print("-------------------------------------------------")
    print("-------------- SAVE TRAINED MODEL ---------------")
    print("-------------------------------------------------")
    print("Saving trained model...")
    save_trained_model(model=model, model_prefix=avail_models[user_choice], path=config.PATH_TO_SAVE_TRAINED_MODEL) # save trained model