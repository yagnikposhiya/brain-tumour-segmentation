"""
author: Yagnik Poshiya
github: @yagnikposhiya
organization: NEUROWORK Research Labs
"""

import os
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F

from torch import nn

class UNet(pl.LightningModule):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        # Encoder
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Final output layer
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ) # return convolutional block
    
    def forward(self, x):
        
        # This forward function specifies how the input data passes through the network's layers and operations to produce the output.
        # Forward function takes tensor or tensors as input and returns tensor or tensors as output.
        # Forward function actually decide flow, how the data or input passes through the neural network.
        
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        # Final output layer
        out = self.out_conv(dec1)
        return out

    def training_step(self, batch, batch_idx):
        images, masks = batch # load input images and mask images from a single-single batch
        outputs = self(images) # calculate the prediction

        """
        EXPLANATION ON preds = torch.argmax(outputs, dim=1):
        outputs: This tensor represents the raw predictions from the model. 
        For multi-class segmentation, its shape is typically (N,C,H,W) where:

        N is the batch size.
        C is the number of classes.
        H is the height of the image.
        W is the width of the image.

        torch.argmax(outputs, dim=1): This function finds the index of the maximum value along the specified dimension, 
        which in this case is dim=1 (the class dimension). 
        Essentially, for each pixel, it selects the class with the highest predicted probability.

        preds: This tensor represents the predicted class labels for each pixel in the input images. After applying torch.argmax, 
        the shape of preds will be (N,H,W) where each value corresponds to the predicted class for that pixel.
        """

        # compute metrics
        preds = torch.argmax(outputs, dim=1) # convert raw outputs to predicted class labels
        loss = F.cross_entropy(outputs, masks) # calculate the cross-entropy loss
        dice = self.dice_coefficient(preds,masks) # calculate dice coefficient
        jaccard = self.jaccard_score(preds,masks) # calculate jaccard score
        sensitivity = self.sensitivity(preds,masks) # calculate sensitivity
        specificity = self.specificity(preds,masks) # calculate specificity

        # log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the loss logs for visualization
        self.log('train_dice', dice, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the dice logs for visualization
        self.log('train_jaccard', jaccard, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the jaccard logs for visualization
        self.log('train_sensitivity', sensitivity, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the sensitivity logs for visualization
        self.log('train_specificity', specificity, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the specificity logs for visualization

        return loss
    
    def dice_coefficient(self, preds, targets, smooth=1):
        preds = preds.contiguous() # contiguous() method to ensure that both preds and targets tensors are stored in a contiguous block of memory. 
                                # This ensures that subsequent operations on these tensors are efficient and error-free.
        targets = targets.contiguous()
        intersection = (preds * targets).sum(dim=2).sum(dim=1)
        dice = (2. * intersection + smooth) / (preds.sum(dim=2).sum(dim=1) + targets.sum(dim=2).sum(dim=1) + smooth)
        return dice.mean()
    
    def jaccard_score(self, preds, targets, smooth=1):
        preds = preds.contiguous()
        targets = targets.contiguous()
        intersection = (preds*targets).sum(dim=2).sum(dim=1)
        union = preds.sum(dim=2).sum(dim=1) + targets.sum(dim=2).sum(dim=1) - intersection
        jaccard = (intersection + smooth) / (union + smooth)
        return  jaccard.mean()
    
    def sensitivity(self, preds, targets, smooth=1):
        preds = preds.contiguous()
        targets = targets.contiguous()
        true_positive = (preds * targets).sum(dim=2).sum(dim=1)
        false_negative = (targets * (1 - preds)).sum(dim=2).sum(dim=1)
        sensitivity = (true_positive + smooth) / (true_positive + false_negative + smooth)
        return sensitivity.mean()
    
    def specificity(self, preds, targets, smooth=1):
        preds = preds.contiguous()
        targets = targets.contiguous()
        true_negative = ((1 - preds) * (1 - targets)).sum(dim=2).sum(dim=1)
        false_positive = ((1 - targets) * preds).sum(dim=2).sum(dim=1)
        specificity = (true_negative + smooth) / (true_negative + false_positive + smooth)
        return specificity.mean()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001) # set optimizer and learning_rate

