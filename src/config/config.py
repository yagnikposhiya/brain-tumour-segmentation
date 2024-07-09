"""
author: Yagnik Poshiya
github: @yagnikposhiya
organization: Neurowork Research Labs
"""

import os

class Config():
    def __init__(self) -> None:
        
        # current working directory
        self.CWD = os.getcwd() # get current working directory

        # training and validation dataset path before storing dataset into .npy format
        self.TRAINSET_PATH = os.path.join(self.CWD,'data/raw/') # set training directory path
        self.PATH_TO_SAVE_PROCESSED_DATA = os.path.join(self.CWD,'data/') # set training directory path
        self.VALIDSET_PATH = os.path.join(self.CWD,'data/raw/') # set validation directory path
        self.TRAIN_IMAGE_PATH =['unstructured/BraTS20_Training_009_flair.nii',
                                'unstructured/BraTS20_Training_009_t1.nii',
                                'unstructured/BraTS20_Training_009_t1ce.nii',
                                'unstructured/BraTS20_Training_009_t2.nii',
                                'unstructured/BraTS20_Training_009_seg.nii'] # set path for any images available in the train directory
        
        # training and validation dataset path after storing dataset into .npy format
        self.TRAIN_IMAGES_DIR = '' # automatically set path for the directory contains stacked images in .npy format (training)
        self.TRAIN_MASKS_DIR = '' # automatically set path for the directory contains mask images in .npy format (training)
        self.VAL_IMAGES_DIR = '' # set path for the directory contains stacked images in .npy format (validation)
        self.VAL_MASKS_DIR = '' # set path for the directory contains mask images in .npy format (validation)
        self.PATH_TO_SAVE_TRAINED_MODEL = os.path.join(self.CWD,'saved_models/') # set path to save trained model

        # model training parameters
        self.BATCH_SIZE = 16 # set batch size for model training
        self.MAX_EPOCHS = 2 # set maximum epochs for model training
        self.NUM_CLASSES = 4 # set number of classes for contains by mask images; here [0,1,2,3]
        self.LEARNING_RATE =0.001 # set learning rate
        self.TRANSFORM = True # set boolean value for applying augmentation techniques for training set and techniques are horizontal flip and vertical flip

        # weights and biases config
        self.ENTITY = 'neuralninjas' # set team/organization name for wandb account
        self.PROJECT = 'NRL-brain-tumour-segmentation' # set project name
        self.REINIT = True # set boolean value for reinitialization
        self.ANONYMOUS = 'allow' # set anonymous value type
        self.LOG_MODEL = 'all' # set log model type

    def printConfiguration(self):
        """
        This function is used to print all configuration related to paths and model training params.

        Parameters:
        - (None)

        Returns:
        - (None)
        """

        print(f"Configurations:")
        print(f"Current Working Directory: {self.CWD}, Trainset_path: {self.TRAINSET_PATH}, Validset_path: {self.VALIDSET_PATH}, ",
              f"Train_image_path: {self.TRAIN_IMAGE_PATH}, Path_to_save_processed_data: {self.PATH_TO_SAVE_PROCESSED_DATA}, ",
              f"Path_to_save_trained_model: {self.PATH_TO_SAVE_TRAINED_MODEL}, Batch_size: {self.BATCH_SIZE}, Max_epochs: {self.MAX_EPOCHS}, ",
              f"Num_classes: {self.NUM_CLASSES}, Learning_rate: {self.LEARNING_RATE}, Trasform/Data_augmentation: {self.TRANSFORM}")
        