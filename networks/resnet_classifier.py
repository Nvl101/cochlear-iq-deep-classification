'''
resnet 18 classifier for 3-class image quality classification
1. modify network from resnet18
2. define loss function
3. define optimizer
'''
import torch
from torch.nn import Linear, Conv2d, Sequential
from torchvision import models


# downsampling block
class DownSampling_Block(Sequential):
    '''
    downsampling via a sequence of layers
    '''
    def __init__(self, **kwargs):
        '''
        arguments:
            - input_size: size of the input 
            - output_size: size of the output
            - kernel_size: size of the convolution filter
        '''
        # resolving the arguments
        self.conv1 = Conv2d(1, 32, kernel_size=3, stride=2, padding=1) 
        self.conv2 = Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        '''
        forward propagation
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class ResnetClassifier(torch.nn.Sequential):
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


# standard resnet18 classifier with input size of 64
resnet18_classifier = models.resnet18(weights='IMAGENET1K_V1')
# modify the conv1 layer to support grayscale in place of RGB
# resnet18_classifier.conv1 = Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), \
#     padding=(3, 3), bias=False, dtype=torch.float32)
# modify output layer into a 3-class classifier
# resnet18_classifier.fc = Linear(in_features=512, out_features=3, bias=True)
resnet18_classifier.fc = ResnetClassifier(in_channels=512, n_classes=3)

# a deeper resnet
resnet50_classifier = models.resnet50(weights='IMAGENET1K_V1')
resnet50_classifier.fc = ResnetClassifier(in_channels=2048, n_classes=3)
# resnet50_classifier.fc = torch.nn.Linear(in_features=2048, out_features=3, bias=True)

resnet101_classifier = models.resnet101(weights='IMAGENET1K_V1')
resnet101_classifier.fc = ResnetClassifier(in_channels=2048, n_classes=3)

# widened resnet18 classifier with input size of 128
# resnet18_128_classifier = models.resnet18()
# resnet18_128_classifier.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7,7), \
#                                                 stride=(4,4), padding=(3,3),bias=False)
# resnet18_128_classifier.fc = Linear(in_features=512, out_features=3, bias=True)


# NOTE: the loss function and optimizer are currently defined in training.py
learning_rate = 0.00005
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet18_classifier.parameters(), lr=learning_rate)

if __name__ == '__main__':
    print("Debug...")
