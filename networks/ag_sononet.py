'''
attention-gated sononet classifier
'''
import torch
import torch.nn as nn


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True):
        super().__init__()

        self.cbr_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),
                                 nn.ReLU(inplace=True),)
    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs
    
class conv2dBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True):
        super().__init__()

        self.cb_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),)
    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super().__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            # kaiming init weight for conv and linear
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            # hardcode weights for batchnorm
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal(m.weight.data, 1.0, 0.02)
                nn.init.constant(m.bias.data, 0.0)

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)

        return x

class Sononet(nn.Module):
    '''
    the original sononet
    '''
    def __init__(self, feature_scale=4, n_classes=3, in_channels=1, is_batchnorm=True, n_convs=None):
        super().__init__()
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_classes= n_classes

        filters = [64, 128, 256, 512]
        filters = [int(x / self.feature_scale) for x in filters]

        if n_convs is None:
            n_convs = [2,2,3,3,3]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm, n=n_convs[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm, n=n_convs[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm, n=n_convs[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm, n=n_convs[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[3], self.is_batchnorm, n=n_convs[4])

        # adaptation layer
        self.conv5_p = conv2DBatchNormRelu(filters[3], filters[2], 1, 1, 0)
        self.conv6_p = conv2dBatchNorm(filters[2], self.n_classes, 1, 1, 0)

        # initialise weights with kaiming
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            # hardcode weights for batchnorm
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal(m.weight.data, 1.0, 0.02)
                nn.init.constant(m.bias.data, 0.0)

    def forward(self, inputs):
        # Feature Extraction
        conv1    = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        conv2    = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3    = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4    = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        conv5    = self.conv5(maxpool4)
        conv5_p  = self.conv5_p(conv5)
        conv6_p  = self.conv6_p(conv5_p)

        batch_size = inputs.shape[0]
        pooled     = nn.functional.adaptive_avg_pool2d(conv6_p, (1, 1)).view(batch_size, -1)    

        return pooled

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = nn.functional.softmax(pred, dim=1)
        return log_p

# class AttentionGatedSononet(torch.nn.Module):
#     '''
#     sononet classifier with attention gates
#     '''

if __name__ == '__main__':
    sononet = Sononet()
    print('debug...')
