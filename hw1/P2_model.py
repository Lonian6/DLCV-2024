import os.path as osp
# import fcn
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
# from torchvision import models
# import logging
# import os
# import time

class fcn32(nn.Module):
    def __init__(self, n_class, pre_trained_vgg):
        super(fcn32, self).__init__()
        #卷积层使用VGG16的
        self.features = pre_trained_vgg.features
        #将全连接层替换成卷积层
        self.conv1 = nn.Conv2d(512, 4096, 1)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout2d()
        
        self.conv2 = nn.Conv2d(4096, n_class, 1)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout2d()
        #上采样，这里只用到了32的
        # self.upsample2x = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False)
        # self.upsample8x = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.upsample32x = nn.Upsample(scale_factor=32,mode='bilinear',align_corners=False)
        
    def forward(self, x):
        s = self.features(x)
        s = self.conv1(s)
        s = self.relu1(s)
        s = self.drop1(s)
        s = self.conv2(s)
        s = self.relu2(s)
        s = self.drop2(s)
        s = self.upsample32x(s)
        return s

class fcn8(nn.Module):
    def __init__(self, n_class, pre_trained_vgg):
        super(fcn8, self).__init__()
        self.num_classes = n_class # 分隔的总类别
        self.backbone = pre_trained_vgg.features # 下采样使用vgg19.features
        # 定义上采样的层操作
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # self.drop3 = nn.Dropout2d()
        self.deconv4 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        # self.drop4 = nn.Dropout2d()
        self.deconv5 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        # self.drop5 = nn.Dropout2d()
        
        self.classifier = nn.Sequential(
                                nn.Conv2d(32, self.num_classes, 1),
                                nn.ReLU(inplace=True),
                                nn.Dropout2d()
        )
        
        # self.classifier = nn.Sequential(
        #                     nn.Conv2d(32, 512, 1),
        #                     nn.ReLU(inplace=True),
        #                     nn.Dropout2d(),
        #                     nn.Conv2d(512, n_class, 1),
        #                     nn.ReLU(inplace=True),
        #                     nn.Dropout2d()
        #                 )
        
        
        # VGG19的MaxPooling所在层，用于逐点相加
        self.pooling_layers = {'4': 'maxpool_1', '9': 'maxpool_2', '18': 'maxpool_3', '27': 'maxpool_4', '36': 'maxpool_5'}

    def forward(self, x):
        output = {}
        # 对图像做下采样，并hook出pooling层特征
        for name, layer in self.backbone._modules.items():
            # 从第一层开始获取图像下采样特征
            x = layer(x)
            # 如果是pooling层，则保存到output中
            if name in self.pooling_layers:
                output[self.pooling_layers[name]] = x
        P5 = output['maxpool_5'] # size=(N, 512, x.H/32, x.W/32)
        P4 = output['maxpool_4'] # size=(N, 512, x.H/16, x.W/16)
        P3 = output['maxpool_3'] # size=(N, 512, x.H/8, x.W/8)
        # 对特征做转置卷积，即上采样，放大到原来大小
        T5 = self.relu(self.deconv1(P5)) # size=(N, 512, x.H/16, x.W/16)
        T5 = self.bn1(T5 + P4) # 特征逐点相加
        T4 = self.relu(self.deconv2(T5)) # size=(N, 256, x.H/8, x.W/8)
        T4 = self.bn2(T4 + P3)
        
        T3 = self.bn3(self.relu(self.deconv3(T4))) # size=(N, 128, x.H/4, x.W/4)
        T2 = self.bn4(self.relu(self.deconv4(T3))) # size=(N, 64, x.H/2, x.W/2)
        T1 = self.bn5(self.relu(self.deconv5(T2))) # size=(N, 32, x.H, x.W)
        
        # T3 = self.drop3(self.relu(self.bn3(self.deconv3(T4)))) # size=(N, 128, x.H/4, x.W/4)
        # T2 = self.drop4(self.relu(self.bn4(self.deconv4(T3)))) # size=(N, 64, x.H/2, x.W/2)
        # T1 = self.drop5(self.relu(self.bn5(self.deconv5(T2)))) # size=(N, 32, x.H, x.W)
        
        
        
        score = self.classifier(T1) # 最后一层卷积输出, size=(N, num_classes, x.H, x.W)
        return score
