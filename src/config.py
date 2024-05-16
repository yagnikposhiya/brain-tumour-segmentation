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

        # training and validation dataset path
        self.TRAINSET_PATH = os.path.join(self.CWD,'data/raw/') # set training directory path
        self.VALIDSET_PATH = '' # set validation directory path
        self.TRAIN_IMAGE_PATH =['unstructured/BraTS20_Training_001_flair.nii',
                                'unstructured/BraTS20_Training_001_t1.nii',
                                'unstructured/BraTS20_Training_001_t1ce.nii',
                                'unstructured/BraTS20_Training_001_t2.nii',
                                'unstructured/BraTS20_Training_001_seg.nii'] # set path for any images available in the train directory
        