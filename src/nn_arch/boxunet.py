"""
author: Yagnik Poshiya
github: @yagnikposhiya
organization: Neurowork Research Labs
"""

import os
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F

from torch import nn

class BoxUNet(pl.LightningModule):
    def __init__(self, num_classes, learning_rate, optimizer):

        self.lr = learning_rate # set learning rate
        self.num_classes = num_classes # set output segmentation classes
        self.optimizer = optimizer # set optimizer
        super(BoxUNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(4, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(768, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(512, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(320, 64)

        # optimized way to implement boxunet architecture
        # create upsample blocks which are core component of boxes
        self.upsample_512_256 = self.upsample_block(512,256)
        self.upsample_256_128 = self.upsample_block(256,128)
        self.upsample_128_64 = self.upsample_block(128,64)

        # manual way to implement boxunet architecture
        # self.upsample1 = self.upsample_block(512,256)
        # self.upsample2 = self.upsample_block(512,256)
        # self.upsample3 = self.upsample_block(256,128)
        # self.upsample4 = self.upsample_block(512,256)
        # self.upsample5 = self.upsample_block(256,128)
        # self.upsample6 = self.upsample_block(128,64)
        # self.upsample7 = self.upsample_block(256,128)
        # self.upsample8 = self.upsample_block(256,128)
        # self.upsample9 = self.upsample_block(128,64)
        # self.upsample10 = self.upsample_block(128,64)
        
        # Final output layer
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ) # return convolutional block
    
    def upsample_block(self,in_channels,out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True)
        ) # return upsampple block which is core component of each and every box in the boxunet architecture
    
    def forward(self, x):
        
        # This forward function specifies how the input data passes through the network's layers and operations to produce the output.
        # Forward function takes tensor or tensors as input and returns tensor or tensors as output.
        # Forward function actually decide flow, how the data or input passes through the neural network.
        
        # print(f"- Shape input image: {x.shape}")

        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        # print(f"- Shape enc4: {enc4.shape}")

        # link all the core components of boxes and create boxes
        upsample1 = self.upsample_512_256(enc4) # box-1

        upsample2 = self.upsample_512_256(enc4) # box-2
        upsample3 = self.upsample_256_128(upsample2) # box-2

        upsample4 = self.upsample_512_256(enc4) # box-3
        upsample5 = self.upsample_256_128(upsample4) # box-3
        upsample6 = self.upsample_128_64(upsample5) # box-3

        upsample7 = self.upsample_256_128(enc3) # box-4

        upsample8 = self.upsample_256_128(enc3) # box-5
        upsample9 = self.upsample_128_64(upsample8) # box-5

        upsample10 = self.upsample_128_64(enc2) # box-6


        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        # print(f"- Shape dec3: {dec3.shape}")
        # print(f"- Shappe enc3: {enc3.shape}")
        # print(f"- Shape upsample1: {upsample1.shape}")
        dec3 = torch.cat((dec3, enc3, upsample1), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2, upsample3, upsample7), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1, upsample6, upsample9, upsample10), dim=1)
        dec1 = self.dec1(dec1)
        
        # Final output layer
        out = self.out_conv(dec1)
        # print(f"- Shape output image: {out.shape}")

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

        (dice_mean_over_classes,dice_label_1,dice_label_2,dice_label_3) = self.dice_coefficient(preds,masks) # calculate dice coefficient and take mean over batch
        dice_mean_over_batch = dice_mean_over_classes.mean() # take mean over batch
        dice_label_1 = dice_label_1.mean() # label-1 wise mean
        dice_label_2 = dice_label_2.mean() # label-2 wise mean
        dice_label_3 = dice_label_3.mean() # label-3 wise mean

        (jaccard_mean_over_classes,jaccard_label_1,jaccard_label_2,jaccard_label_3) = self.jaccard_score(preds,masks) # calculate jaccard score and take mean over batch
        jaccard_mean_over_batch = jaccard_mean_over_classes.mean() # take mean over batch
        jaccard_label_1 = jaccard_label_1.mean() # label-1 wise mean
        jaccard_label_2 = jaccard_label_2.mean() # label-2 wise mean
        jaccard_label_3 = jaccard_label_3.mean() # label-3 wise mean

        (sensitivity_mean_over_classes,sensitivity_label_1,sensitivity_label_2,sensitivity_label_3) = self.sensitivity(preds,masks) # calculate sensitivity and take mean over batch
        sensitivity_mean_over_batch = sensitivity_mean_over_classes.mean() # take mean over batch
        sensitivity_label_1 = sensitivity_label_1.mean() # label-1 wise mean
        sensitivity_label_2 = sensitivity_label_2.mean() # label-2 wise mean
        sensitivity_label_3 = sensitivity_label_3.mean() # label-3 wise mean


        (specificity_mean_over_classes,specificity_label_1,specificity_label_2,specificity_label_3) = self.specificity(preds,masks) # calculate specificity and take mean over batch
        specificity_mean_over_batch = specificity_mean_over_classes.mean() # take mean over batch
        specificity_label_1 = specificity_label_1.mean() # label-1 wise mean
        specificity_label_2 = specificity_label_2.mean() # label-2 wise mean
        specificity_label_3 = specificity_label_3.mean() # label-3 wise mean

        # log metrics
        self.log('boxunet_train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the loss logs for visualization

        self.log('boxunet_train_dice_mean_over_batch', dice_mean_over_batch, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the dice logs for visualization
        self.log('boxunet_train_dice_mean_Necrotic_Core', dice_label_1, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the dice_label_1 logs for visualization
        self.log('boxunet_train_dice_mean_Peritumoral_Edema', dice_label_2, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the dice_label_2 logs for visualization
        self.log('boxunet_train_dice_mean_GDEnhancing_Tumor', dice_label_3, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the dice_label_3 logs for visualization

        self.log('boxunet_train_jaccard_mean_over_batch', jaccard_mean_over_batch, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the jaccard logs for visualization
        self.log('boxunet_train_jaccard_mean_Necrotic_Core', jaccard_label_1, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the jaccard_label_1 logs for visualization
        self.log('boxunet_train_jaccard_mean_Peritumoral_Edema', jaccard_label_2, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the jaccard_label_2 logs for visualization
        self.log('boxunet_train_jaccard_mean_GDEnhancing_Tumor', jaccard_label_3, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the jaccard_label_3 logs for visualization

        self.log('boxunet_train_sensitivity_mean_over_batch', sensitivity_mean_over_batch, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the sensitivity logs for visualization
        self.log('boxunet_train_sensitivity_mean_Necrotic_Core', sensitivity_label_1, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the sensitivity_label_1 logs for visualization
        self.log('boxunet_train_sensitivity_mean_Peritumoral_Edema', sensitivity_label_2, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the sensitivity_label_2 logs for visualization
        self.log('boxunet_train_sensitivity_mean_GDEnhancing_Tumor', sensitivity_label_3, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the sensitivity_label_3 logs for visualization


        self.log('boxunet_train_specificity_mean_over_batch', specificity_mean_over_batch, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the specificity logs for visualization
        self.log('boxunet_train_specificity_mean_Necrotic_Core', specificity_label_1, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the specificity_label_1 logs for visualization
        self.log('boxunet_train_specificity_mean_Peritumoral_Edema', specificity_label_2, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the specificity_label_2 logs for visualization
        self.log('boxunet_train_specificity_mean_GDEnhancing_Tumor', specificity_label_3, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the specificity_label_2 logs for visualization

        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch # load input images and mask images from a single-single batch
        outputs = self(images) # calculate the prediction

        # compute metrics
        preds = torch.argmax(outputs, dim=1) # convert raw outputs to predicted class labels
        loss = F.cross_entropy(outputs, masks) # calculate the cross-entropy loss

        (dice_mean_over_classes,dice_label_1,dice_label_2,dice_label_3) = self.dice_coefficient(preds,masks) # calculate dice coefficient and take mean over batch
        dice_mean_over_batch = dice_mean_over_classes.mean() # take mean over batch
        dice_label_1 = dice_label_1.mean() # label-1 wise mean
        dice_label_2 = dice_label_2.mean() # label-2 wise mean
        dice_label_3 = dice_label_3.mean() # label-3 wise mean

        (jaccard_mean_over_classes,jaccard_label_1,jaccard_label_2,jaccard_label_3) = self.jaccard_score(preds,masks) # calculate jaccard score and take mean over batch
        jaccard_mean_over_batch = jaccard_mean_over_classes.mean() # take mean over batch
        jaccard_label_1 = jaccard_label_1.mean() # label-1 wise mean
        jaccard_label_2 = jaccard_label_2.mean() # label-2 wise mean
        jaccard_label_3 = jaccard_label_3.mean() # label-3 wise mean

        (sensitivity_mean_over_classes,sensitivity_label_1,sensitivity_label_2,sensitivity_label_3) = self.sensitivity(preds,masks) # calculate sensitivity and take mean over batch
        sensitivity_mean_over_batch = sensitivity_mean_over_classes.mean() # take mean over batch
        sensitivity_label_1 = sensitivity_label_1.mean() # label-1 wise mean
        sensitivity_label_2 = sensitivity_label_2.mean() # label-2 wise mean
        sensitivity_label_3 = sensitivity_label_3.mean() # label-3 wise mean


        (specificity_mean_over_classes,specificity_label_1,specificity_label_2,specificity_label_3) = self.specificity(preds,masks) # calculate specificity and take mean over batch
        specificity_mean_over_batch = specificity_mean_over_classes.mean() # take mean over batch
        specificity_label_1 = specificity_label_1.mean() # label-1 wise mean
        specificity_label_2 = specificity_label_2.mean() # label-2 wise mean
        specificity_label_3 = specificity_label_3.mean() # label-3 wise mean

        # log metrics
        self.log('boxunet_valid_loss', loss, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the loss logs for visualization

        self.log('boxunet_valid_dice_mean_over_batch', dice_mean_over_batch, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the dice logs for visualization
        self.log('boxunet_valid_dice_mean_Necrotic_Core', dice_label_1, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the dice_label_1 logs for visualization
        self.log('boxunet_valid_dice_mean_Peritumoral_Edema', dice_label_2, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the dice_label_2 logs for visualization
        self.log('boxunet_valid_dice_mean_GDEnhancing_Tumor', dice_label_3, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the dice_label_3 logs for visualization

        self.log('boxunet_valid_jaccard_mean_over_batch', jaccard_mean_over_batch, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the jaccard logs for visualization
        self.log('boxunet_valid_jaccard_mean_Necrotic_Core', jaccard_label_1, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the jaccard_label_1 logs for visualization
        self.log('boxunet_valid_jaccard_mean_Peritumoral_Edema', jaccard_label_2, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the jaccard_label_2 logs for visualization
        self.log('boxunet_valid_jaccard_mean_GDEnhancing_Tumor', jaccard_label_3, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the jaccard_label_3 logs for visualization

        self.log('boxunet_valid_sensitivity_mean_over_batch', sensitivity_mean_over_batch, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the sensitivity logs for visualization
        self.log('boxunet_valid_sensitivity_mean_Necrotic_Core', sensitivity_label_1, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the sensitivity_label_1 logs for visualization
        self.log('boxunet_valid_sensitivity_mean_Peritumoral_Edema', sensitivity_label_2, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the sensitivity_label_2 logs for visualization
        self.log('boxunet_valid_sensitivity_mean_GDEnhancing_Tumor', sensitivity_label_3, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the sensitivity_label_3 logs for visualization

        self.log('boxunet_valid_specificity_mean_over_batch', specificity_mean_over_batch, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the specificity logs for visualization
        self.log('boxunet_valid_specificity_mean_Necrotic_Core', specificity_label_1, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the specificity_label_1 logs for visualization
        self.log('boxunet_valid_specificity_mean_Peritumoral_Edema', specificity_label_2, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the specificity_label_2 logs for visualization
        self.log('boxunet_valid_specificity_mean_GDEnhancing_Tumor', specificity_label_3, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the specificity_label_3 logs for visualization

        return loss
    
    def dice_coefficient(self, preds, targets, smooth=1):
        """
        Classes and their labels:
        label-0: Background
        label-1: Necrotic and Non-enhancing Tumor Core (NCR/NET)
        label-2: Peritumoral Edema
        label-3: GD-enhancing Tumor
        """
        preds = F.one_hot(preds, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        targets = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        intersection = (preds * targets).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice.mean(dim=1), dice[:,1], dice[:,2], dice[:3] # (mean over classes, label-1, label-2, label-3)
    
    def jaccard_score(self, preds, targets, smooth=1):
        preds = F.one_hot(preds, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        targets = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        intersection = (preds * targets).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection
        jaccard = (intersection + smooth) / (union + smooth)
        return  jaccard.mean(dim=1), jaccard[:,1], jaccard[:,2], jaccard[:3] # (mean over classes, label-1, label-2, label-3)
    
    def sensitivity(self, preds, targets, smooth=1):
        preds = F.one_hot(preds, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        targets = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        true_positive = (preds * targets).sum(dim=(2, 3))
        false_negative = (targets * (1 - preds)).sum(dim=(2, 3))
        sensitivity = (true_positive + smooth) / (true_positive + false_negative + smooth)
        return sensitivity.mean(dim=1), sensitivity[:,1], sensitivity[:,2], sensitivity[:3] # (mean over classes, label-1, label-2, label-3)
    
    def specificity(self, preds, targets, smooth=1):
        preds = F.one_hot(preds, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        targets = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        true_negative = ((1 - preds) * (1 - targets)).sum(dim=(2, 3))
        false_positive = ((1 - targets) * preds).sum(dim=(2, 3))
        specificity = (true_negative + smooth) / (true_negative + false_positive + smooth)
        return specificity.mean(dim=1), specificity[:,1], specificity[:,2], specificity[:3] # (mean over classes, label-1, label-2, label-3)

    def configure_optimizers(self):
        if self.optimizer == 'Adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr) # set optimizer and learning_rate
        elif self.optimizer == 'AdamW':
            return torch.optim.AdamW(self.parameters(),lr=self.lr) # set optimizer and learning rate
        elif self.optimizer == 'RMSProp':
            return torch.optim.RMSprop(self.parameters(), lr=self.lr) # set optimizer and learning rate
        elif self.optimizer == 'SGD':
            return torch.optim.SGD(self.parameters(), lr=self.lr) # set optimizer snd learning rate
