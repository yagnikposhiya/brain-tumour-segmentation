"""
author: Yagnik Poshiya
github: @yagnikposhiya
organization: NEUROWORK Research Labs
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

"""
In-Place Operation in ReLU:
[1] In-Place Operation: When inplace=True is used, the ReLU operation is performed directly on the input tensor, modifying 
    its values without allocating additional memory for the output. This can save memory and potentially improve performance.

[2] Standard Operation: When inplace=False, the ReLU operation creates a new tensor with the result, 
    leaving the input tensor unchanged.
"""

class HSwish(nn.Module):
    def forward(self,x):
        return x * F.relu6(x+3,inplace=True) / 6
    
class SEBlock(nn.Module):
    def __init__(self,in_channels,reduction=4) -> None:
        super(SEBlock,self).__init__()

        self.fc1 = nn.Conv2d(in_channels, in_channels //reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self,x):
        scale = torch.mean(x, dim=(2,3), keepdim=True)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale))

        return x * scale
    
class InvertedResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,expand_ratio,use_se=True,activation='HS') -> None:
        super(InvertedResidualBlock,self).__init__()

        self.stride = stride # set stride value
        self.use_res_connect = (self.stride == 1 and in_channels == out_channels) # set boolean value for creating residual connection
        hidden_dim = int(in_channels * expand_ratio) # set hidden dimension

        if activation == 'HS':
            activation_layer = HSwish()
        else:
            activation_layer = nn.ReLU(inplace=True)

        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels,hidden_dim,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(hidden_dim),
            activation_layer
        )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim,hidden_dim,kernel_size=kernel_size,stride=stride,padding=kernel_size//2,groups=hidden_dim,bias=False),
            nn.BatchNorm2d(hidden_dim),
            activation_layer
        )

        self.se = SEBlock(hidden_dim) if use_se else nn.Identity()

        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim,out_channels,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self,x):
        identity = x
        out = self.expand_conv(x)
        out = self.depthwise_conv(out)
        out = self.se(out)
        out = self.project_conv(out)

        if self.use_res_connect:
            out += identity
        
        return out
    
class MobileNetV3LargeUNet(pl.LightningModule):
    def __init__(self,num_classes,learning_rate) -> None:
        self.lr = learning_rate # set learning rate
        self.num_classes = num_classes # set output segmentation classes
        super(MobileNetV3LargeUNet,self).__init__()

        # encoder (mobilenetv3-small) input layer
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(16),
            HSwish()
        )

        # encoder Block-1
        self.output_block_11 = InvertedResidualBlock(in_channels=16,out_channels=16,kernel_size=3,stride=1,expand_ratio=1,use_se=True,activation='RE')

        # encoder Block-2
        self.output_block_21 = InvertedResidualBlock(in_channels=16,out_channels=24,kernel_size=3,stride=2,expand_ratio=4,use_se=False,activation='RE')
        self.output_block_22 = InvertedResidualBlock(in_channels=24,out_channels=24,kernel_size=3,stride=1,expand_ratio=3,use_se=False,activation='RE')

        # encoder Block-3
        self.output_block_31 = InvertedResidualBlock(in_channels=24,out_channels=40,kernel_size=5,stride=2,expand_ratio=3,use_se=True,activation='RE')
        self.output_block_32 = InvertedResidualBlock(in_channels=40,out_channels=40,kernel_size=5,stride=1,expand_ratio=3,use_se=True,activation='RE')
        self.output_block_33 = InvertedResidualBlock(in_channels=40,out_channels=40,kernel_size=5,stride=1,expand_ratio=3,use_se=True,activation='RE')

        # encoder Block-4
        self.output_block_41 = InvertedResidualBlock(in_channels=40,out_channels=80,kernel_size=3,stride=2,expand_ratio=6,use_se=False,activation='HS')
        self.output_block_42 = InvertedResidualBlock(in_channels=80,out_channels=80,kernel_size=3,stride=1,expand_ratio=2.5,use_se=False,activation='HS')
        self.output_block_43 = InvertedResidualBlock(in_channels=80,out_channels=80,kernel_size=3,stride=1,expand_ratio=2.3,use_se=False,activation='HS')
        self.output_block_44 = InvertedResidualBlock(in_channels=80,out_channels=80,kernel_size=3,stride=1,expand_ratio=2.3,use_se=False,activation='HS')

        # encoder Block-5
        self.output_block_51 = InvertedResidualBlock(in_channels=80,out_channels=112,kernel_size=3,stride=1,expand_ratio=6,use_se=True,activation='HS')
        self.output_block_52 = InvertedResidualBlock(in_channels=112,out_channels=112,kernel_size=5,stride=1,expand_ratio=6,use_se=True,activation='HS')

        # encoder Block-6
        self.output_block_61 = InvertedResidualBlock(in_channels=112,out_channels=160,kernel_size=5,stride=2,expand_ratio=6,use_se=True,activation='HS')
        self.output_block_62 = InvertedResidualBlock(in_channels=160,out_channels=160,kernel_size=5,stride=1,expand_ratio=6,use_se=True,activation='HS')
        self.output_block_63 = InvertedResidualBlock(in_channels=160,out_channels=160,kernel_size=5,stride=1,expand_ratio=6,use_se=True,activation='HS')

        # bottleneck
        self.bottleneck = self.conv_block(160,320)

        # decoder
        self.upconv6 = nn.ConvTranspose2d(320,160,kernel_size=2,stride=2) # upconv6 + output_block_63
        self.dec6 = self.conv_block(320,160)

        self.upconv5 = nn.ConvTranspose2d(160,112,kernel_size=2,stride=2) # upconv5 + output_block_52
        self.dec5 = self.conv_block(224,112)

        self.upconv4 = nn.ConvTranspose2d(112,80,kernel_size=2,stride=2) # upconv4 + output_block_44
        self.dec4 = self.conv_block(160,80)

        self.upconv3 = nn.ConvTranspose2d(80,40,kernel_size=2,stride=2) # upconv3 + ouptut_block_33
        self.dec3 = self.conv_block(80,40)

        self.upconv2 = nn.ConvTranspose2d(40,24,kernel_size=2,stride=2) # upconv2 + output_block_22
        self.dec2 = self.conv_block(48,24)

        self.upconv1 = nn.ConvTranspose2d(24,16,kernel_size=2,stride=2) # upconv1 + output_block_11
        self.dec1 = self.conv_block(32,16)

        # final output layer
        self.output_layer = nn.Conv2d(16,num_classes,kernel_size=1)


    def conv_block(self,in_channels,out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True)
        ) # return convolutional block
    
    def forward(self,x):

        # input layer
        input_img = self.input_layer(x)

        # encoder
        block_11 = self.output_block_11(input_img)

        block_21 = self.output_block_21(block_11)
        block_22 = self.output_block_22(block_21)

        block_31 = self.output_block_31(block_22)
        block_32 = self.output_block_32(block_31)
        block_33 = self.output_block_33(block_32)

        block_41 = self.output_block_41(block_33)
        block_42 = self.output_block_42(block_41)
        block_43 = self.output_block_43(block_42)
        block_44 = self.output_block_44(block_43)

        block_51 = self.output_block_51(block_44)
        block_52 = self.output_block_52(block_51)

        block_61 = self.output_block_61(block_52)
        block_62 = self.output_block_62(block_61)
        block_63 = self.output_block_63(block_62)

        # bottleneck
        bottleneck = self.bottleneck(block_63)

        # decoder
        upconv6 = self.upconv6(bottleneck)
        block_63_resized = F.interpolate(block_63, size=upconv6.shape[2:], mode='bilinear', align_corners=False)
        upconv6 = torch.cat([upconv6,block_63_resized],dim=1)
        dec6 = self.dec6(upconv6)

        upconv5 = self.upconv5(dec6)
        block_52_resized = F.interpolate(block_52, size=upconv5.shape[2:], mode='bilinear', align_corners=False)
        upconv5 = torch.cat([upconv5,block_52_resized],dim=1)
        dec5 = self.dec5(upconv5)

        upconv4 = self.upconv4(dec5)
        block_44_resized = F.interpolate(block_44, size=upconv4.shape[2:], mode='bilinear', align_corners=False)
        upconv4 = torch.cat([upconv4,block_44_resized],dim=1)
        dec4 = self.dec4(upconv4)

        upconv3 = self.upconv3(dec4)
        block_33_resized = F.interpolate(block_33, size=upconv3.shape[2:], mode='bilinear', align_corners=False)
        upconv3 = torch.cat([upconv3,block_33_resized],dim=1)
        dec3 = self.dec3(upconv3)

        upconv2 = self.upconv2(dec3)
        block_22_resized = F.interpolate(block_22, size=upconv2.shape[2:], mode='bilinear', align_corners=False)
        upconv2 = torch.cat([upconv2,block_22_resized],dim=1)
        dec2 = self.dec2(upconv2)

        upconv1 = self.upconv1(dec2)
        block_11_resized = F.interpolate(block_11, size=upconv1.shape[2:], mode='bilinear', align_corners=False)
        upconv1 = torch.cat([upconv1,block_11_resized],dim=1)
        dec1 = self.dec1(upconv1)

        # final output layer
        out = self.output_layer(dec1)
        out_resized = F.interpolate(out, size=(128,128), mode='bilinear', align_corners=False)

        return out_resized
    
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
        self.log('mobilenetv3_large_train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the loss logs for visualization

        self.log('mobilenetv3_large_train_dice_mean_over_batch', dice_mean_over_batch, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the dice logs for visualization
        self.log('mobilenetv3_large_train_dice_mean_Necrotic_Core', dice_label_1, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the dice_label_1 logs for visualization
        self.log('mobilenetv3_large_train_dice_mean_Peritumoral_Edema', dice_label_2, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the dice_label_2 logs for visualization
        self.log('mobilenetv3_large_train_dice_mean_GDEnhancing_Tumor', dice_label_3, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the dice_label_3 logs for visualization

        self.log('mobilenetv3_large_train_jaccard_mean_over_batch', jaccard_mean_over_batch, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the jaccard logs for visualization
        self.log('mobilenetv3_large_train_jaccard_mean_Necrotic_Core', jaccard_label_1, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the jaccard_label_1 logs for visualization
        self.log('mobilenetv3_large_train_jaccard_mean_Peritumoral_Edema', jaccard_label_2, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the jaccard_label_2 logs for visualization
        self.log('mobilenetv3_large_train_jaccard_mean_GDEnhancing_Tumor', jaccard_label_3, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the jaccard_label_3 logs for visualization

        self.log('mobilenetv3_large_train_sensitivity_mean_over_batch', sensitivity_mean_over_batch, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the sensitivity logs for visualization
        self.log('mobilenetv3_large_train_sensitivity_mean_Necrotic_Core', sensitivity_label_1, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the sensitivity_label_1 logs for visualization
        self.log('mobilenetv3_large_train_sensitivity_mean_Peritumoral_Edema', sensitivity_label_2, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the sensitivity_label_2 logs for visualization
        self.log('mobilenetv3_large_train_sensitivity_mean_GDEnhancing_Tumor', sensitivity_label_3, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the sensitivity_label_3 logs for visualization

        self.log('mobilenetv3_large_train_specificity_mean_over_batch', specificity_mean_over_batch, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the specificity logs for visualization
        self.log('mobilenetv3_large_train_specificity_mean_Necrotic_Core', specificity_label_1, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the specificity_label_1 logs for visualization
        self.log('mobilenetv3_large_train_specificity_mean_Peritumoral_Edema', specificity_label_2, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the specificity_label_2 logs for visualization
        self.log('mobilenetv3_large_train_specificity_mean_GDEnhancing_Tumor', specificity_label_3, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the specificity_label_2 logs for visualization

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
        self.log('mobilenetv3_large_valid_loss', loss, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the loss logs for visualization

        self.log('mobilenetv3_large_valid_dice_mean_over_batch', dice_mean_over_batch, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the dice logs for visualization
        self.log('mobilenetv3_large_valid_dice_mean_Necrotic_Core', dice_label_1, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the dice_label_1 logs for visualization
        self.log('mobilenetv3_large_valid_dice_mean_Peritumoral_Edema', dice_label_2, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the dice_label_2 logs for visualization
        self.log('mobilenetv3_large_valid_dice_mean_GDEnhancing_Tumor', dice_label_3, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the dice_label_3 logs for visualization

        self.log('mobilenetv3_large_valid_jaccard_mean_over_batch', jaccard_mean_over_batch, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the jaccard logs for visualization
        self.log('mobilenetv3_large_valid_jaccard_mean_Necrotic_Core', jaccard_label_1, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the jaccard_label_1 logs for visualization
        self.log('mobilenetv3_large_valid_jaccard_mean_Peritumoral_Edema', jaccard_label_2, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the jaccard_label_2 logs for visualization
        self.log('mobilenetv3_large_valid_jaccard_mean_GDEnhancing_Tumor', jaccard_label_3, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the jaccard_label_3 logs for visualization

        self.log('mobilenetv3_large_valid_sensitivity_mean_over_batch', sensitivity_mean_over_batch, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the sensitivity logs for visualization
        self.log('mobilenetv3_large_valid_sensitivity_mean_Necrotic_Core', sensitivity_label_1, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the sensitivity_label_1 logs for visualization
        self.log('mobilenetv3_large_valid_sensitivity_mean_Peritumoral_Edema', sensitivity_label_2, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the sensitivity_label_2 logs for visualization
        self.log('mobilenetv3_large_valid_sensitivity_mean_GDEnhancing_Tumor', sensitivity_label_3, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the sensitivity_label_3 logs for visualization

        self.log('mobilenetv3_large_valid_specificity_mean_over_batch', specificity_mean_over_batch, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the specificity logs for visualization
        self.log('mobilenetv3_large_valid_specificity_mean_Necrotic_Core', specificity_label_1, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the specificity_label_1 logs for visualization
        self.log('mobilenetv3_large_valid_specificity_mean_Peritumoral_Edema', specificity_label_2, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the specificity_label_2 logs for visualization
        self.log('mobilenetv3_large_valid_specificity_mean_GDEnhancing_Tumor', specificity_label_3, on_step=True, on_epoch=True, prog_bar=True, enable_graph=True) # save the specificity_label_3 logs for visualization

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
        return torch.optim.Adam(self.parameters(), lr=self.lr) # set optimizer and learning_rate
        # return torch.optim.AdamW(self.parameters(),lr=self.lr) # set optimizer and learning rate
    