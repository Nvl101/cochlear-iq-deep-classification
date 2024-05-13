'''
vgg classifier for 3-class image quality classification
'''
import torch
from torchvision import models


class VggClassifier(torch.nn.Module):
    '''
    classifier containing the VGG model and a softmax classification layer
    '''
    def __init__(self, n_classes=3, pretrained: bool = True):
        super().__init__()
        self.net = models.vgg16(weights='IMAGENET1K_V1' if pretrained else 'DEFAULT')
        self.linear_classifier = torch.nn.Linear(1000, n_classes)
        self.softmax_classifier = torch.nn.Softmax(dim=1)
    def forward(self, x):
        x = self.net(x)
        x = self.linear_classifier(x)
        x = self.softmax_classifier(x)

if __name__ == '__main__':
    vgg_classifier = VggClassifier()
    print("debug...")
