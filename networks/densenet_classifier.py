'''
densenet 3-class image quality classification

'''
import torch
import torch.nn as nn
from torch.nn import Sequential, Conv2d, Linear
from torchvision import models
from networks.downsampling import DownSampling_Block


class DensenetClassifier(nn.Sequential):
    '''
    classifier used on densenet, with linear layers followed by softmax layers
    '''
    def __init__(self, in_channels: int = 1024, n_classes: int = 3):
        '''
        inputs:
            in_channels: number of input channels
            n_classes: number of classification classes
            n_linear_layers: number of linear layers before the softmax
        '''
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_features=in_channels, out_features=n_classes, bias=True)
        self.softmax_layer = torch.nn.Softmax(dim=1)
    def forward(self, x):
        x = self.linear_layer(x)
        x = self.softmax_layer(x)
        return x

class DenseNet121_Classifier(Sequential):
    def __init__(self, **kwargs):
        self.downsampling_block = DownSampling_Block()
        self.densenet = models.densenet121()
        self.densenet.classifier = nn.Linear(
            in_features=1024, out_features=3, bias=True)
    def forward(self, x):
        x = self.downsampling_block(x)
        x = self.densenet(x)
        return x

densenet121_classifier = models.densenet121(weights='IMAGENET1K_V1')
# modification: conv0 layer uses 1 channel instead of 3
# densenet121_classifier.conv0 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# modification: 3 output features
# densenet121_classifier.classifier = Linear(in_features=1024, out_features=3, bias=True)
densenet121_classifier.classifier = DensenetClassifier()



class DenseBlock(nn.Module):
    '''
    a single dense block
    '''
    def __init__(self, in_channels: int, growth_rate: int = 12, n_layers: int = 2):
        '''
        inputs:
        - in_channels: number of channels input, 1 for greyscale and 3 for rgb
        - growth_rate: number of features grown after each layer
        '''
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(in_channels + i * growth_rate),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        in_channels + i * growth_rate, growth_rate,
                        kernel_size = 3, stride = 1, padding = 1)
                )
            )

    def forward(self, x):
        for layer in self.layers:
            new_features = layer(x)
            x = torch.cat([x, new_features], dim = 1)
        return x

class TransitionBlock(nn.Module):
    '''
    transition block connecting the dense blocks in network
    '''
    def __init__(self, n_input_features: int, n_output_features: int):
        super().__init__()

class LightDenseNet(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # Assuming grayscale input
        self.denseblock = DenseBlock(32, growth_rate=12, num_layers=3)
        self.pool = nn.AvgPool2d(kernel_size=2) 
        self.classifier = nn.Linear(128, n_classes) # Adjust based on final feature map size

    def forward(self, x):
        x = self.conv1(x)
        x = self.denseblock(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.flatten(x)
        x = self.classifier(x)
        return x 
