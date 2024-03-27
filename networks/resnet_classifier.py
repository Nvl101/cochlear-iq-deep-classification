'''
resnet 18 classifier for 3-class image quality classification
1. modify network from resnet18
2. define loss function
3. define optimizer
'''
import torch
from torch.nn import Linear, Conv2d
from torchvision import models

resnet18_classifier = models.resnet18() #(weights='IMAGENET1K_V1')
# modify the conv1 layer to support grayscale in place of RGB
resnet18_classifier.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), \
    padding=(3, 3), bias=False, dtype=torch.float32)
# modify output layer into a 3-class classifier
resnet18_classifier.fc = Linear(in_features=512, out_features=3, bias=True)

# NOTE: the loss function and optimizer are currently defined in training.py
learning_rate = 0.00005
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet18_classifier.parameters(), lr=learning_rate)
