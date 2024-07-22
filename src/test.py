"""
author: Yagnik Poshiya
github: @yagnikposhiya
organization: Neurowork Research Labs
"""

import torch

from nn_arch.unet import UNet
from config.config import Config
from nn_arch.boxunet import BoxUNet
from nn_arch.mobilenetv1 import MobileNetV1UNet
from nn_arch.mobilenetv2 import MobileNetV2UNet
from nn_arch.mobilenetv3_small import MobileNetV3SmallUNet
from nn_arch.mobilenetv3_large import MobileNetV3LargeUNet
from nn_arch.cascaded_mobilenetv3_large import CascadedMobileNetV3LargeUNet
from nn_arch.mobilenetv3_large_without_SEblock import MobileNetV3LargeUNet_Without_SEBlock
from nn_arch.mobilenetv3_small_without_SEblock import MobileNetV3SmallUNet_Without_SEBlock
from utils.utils import load_saved_model, available_models, prepareImageForInference, groundTruthVSPredicted_AllClasses, available_optimizers, createMontage


if __name__=='__main__':

    config = Config() # create an instance of class Config
        
    print("-------------------------------------------------")
    print("-------------- LOAD TRAINED MODEL ---------------")
    print("-------------------------------------------------")

    avail_models, user_choice = available_models() # get user choice for available models for training
    avail_optim, user_choice_optim = available_optimizers() # get user choice for available optimizers for training
    print(f"- You have selected {avail_optim[user_choice_optim]} optimizer with {avail_models[user_choice]} neural network architecture.")

    if user_choice == 0:
        model_class = UNet # create a normal standard unet model
    elif user_choice == 1:
        model_class = MobileNetV1UNet # create MobileNetV1 model
    elif user_choice == 2:
        model_class = MobileNetV2UNet # create MobileNetV2 model
    elif user_choice == 3:
        model_class = MobileNetV3SmallUNet # create MobileNetV3-Small model
    elif user_choice == 4:
        model_class = MobileNetV3LargeUNet # create MobileNetV3-Large model
    elif user_choice == 5:
        model_class = CascadedMobileNetV3LargeUNet # create Cascaded MobileNetV3-Large model
    elif user_choice == 6:
        model_class = BoxUNet # create BoxUNet model
    elif user_choice == 7:
        model_class = MobileNetV3LargeUNet_Without_SEBlock # create MobileNetV3-Large architecture without SE Block
    elif user_choice == 8:
        model_class = MobileNetV3SmallUNet_Without_SEBlock # create MobileNetV3-Small architecture without SE Block

    loaded_model = load_saved_model(model_class=model_class,num_classes=config.NUM_CLASSES,learning_rate=config.LEARNING_RATE, optimizer=avail_optim[user_choice_optim]) # load saved trained model
    print("Saved model is loaded successfully.")

    print("-------------------------------------------------")
    print("------------------ INFERENCE --------------------")
    print("-------------------------------------------------")

    image, _, np_image, np_mask = prepareImageForInference() # prepare image and mask for inference
    print("Image and mask are prepared successfully.")

    # perform inference
    with torch.no_grad():
        output = loaded_model(image) # performing segmentation
        predicted_mask = torch.argmax(output, dim=1)  # convert raw outputs to predicted class labels
        """
        Applying torch.argmax(output,dim=1):

        It will find the index of the maximum score for each pixel across the class dimension (dim=1).
        This means it will compare the scores across the 4 classes for each pixel and select the class
        with the highest score.

        Let's take a one example:
        For a pixel at position (0,0):
            - Score for class 0: 0.1
            - Score for class 1: 0.3
            - Score for class 2: 0.4
            - Score for class 3: 0.2

        The highest score is 0.4, which corresponds to class 2. So, the predicted class for this pixel will be 2.

        Why Use torch.argmax?

        [1] Converting Continuous Predictions to Discrete Labels:
            The model outputs continuous scores (logits) for each class. torch.argmax converts these scores into discrete 
            class labels by selecting the class with the highest score for each pixel.

        [2] Matching Ground Truth Format:
            Your ground truth masks are integers representing class labels. To compare the model's predictions with 
            the ground truth, you need to convert the model's output into the same format.

        [3] Loss Computation:
            Loss functions like cross-entropy loss expect class labels (not continuous scores) to compute the difference 
            between the predicted and ground truth masks.
        """

    print(f"- Output segmented mask shape: {predicted_mask.shape}")
    print(f"- Output segmented mask unique values: {torch.unique(predicted_mask)}")
    print(f"- Output segmented mask: {predicted_mask}")

    groundTruthVSPredicted_AllClasses(image=np_image, groundtruth_mask=np_mask, predicted_mask=predicted_mask, model=avail_models[user_choice]) # plot the predicted and groundtruth mask

    while True:
        try:
            user_ans = str(input("Do you want to create a montage with multiple architectures and images [Y/N]?: ")) # ask user that if he/she wants to create montage
            if user_ans.lower() == 'y': # if user enters Y/y
                createMontage() # call function to create montage from multiple archs and images
                break # stop loop execution after successful function execution
            else: # otherwise stop loop execution
                break
            
        except ValueError:
            print()