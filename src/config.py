"""
author: Yagnik Poshiya
github: @yagnikposhiya
organization: NEUROWORK Research Labs
"""

import os

class Config():
    def __init__(self) -> None:
        
        # current working directory
        self.CWD = os.getcwd() # get current working directory

        # training and validation dataset path before storing dataset into .npy format
        self.TRAINSET_PATH = os.path.join(self.CWD,'data/raw/') # set training directory path
        self.VALIDSET_PATH = os.path.join(self.CWD,'data/raw/') # set validation directory path
        self.TRAIN_IMAGE_PATH =['unstructured/BraTS20_Training_001_flair.nii',
                                'unstructured/BraTS20_Training_001_t1.nii',
                                'unstructured/BraTS20_Training_001_t1ce.nii',
                                'unstructured/BraTS20_Training_001_t2.nii',
                                'unstructured/BraTS20_Training_001_seg.nii'] # set path for any images available in the train directory
        
        # training and validation dataset path after storing dataset into .npy format
        self.TRAIN_IMAGES_DIR = '' # automatically set path for the directory contains stacked images in .npy format (training)
        self.TRAIN_MASKS_DIR = '' # automatically set path for the directory contains mask images in .npy format (training)
        self.VAL_IMAGES_DIR = '' # set path for the directory contains stacked images in .npy format (validation)
        self.VAL_MASKS_DIR = '' # set path for the directory contains mask images in .npy format (validation)

        # model training parameters
        self.BATCH_SIZE = 16 # set batch size for model training
        self.MAX_EPOCHS = 3 # set maximum epochs for model training
        self.NUM_CLASSES = 4 # set number of classes for contains by mask images; here [0,1,2,3]

        # weights and biases config
        self.ENTITY = 'neuralninjas' # set team/organization name for wandb account
        self.PROJECT = 'NRL-brain-tumour-segmentation' # set project name
        self.REINIT = True # set boolean value for reinitialization
        self.ANONYMOUS = 'allow' # set anonymous value type
        self.GROUP = [
            'Standard-UNet'
        ] # set group name
        self.LOG_MODEL = 'all' # set log model type
        