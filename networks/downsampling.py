'''
downsampling block, to reduce image size to fit the network
'''
from torch.nn import Sequential, Conv2d


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
