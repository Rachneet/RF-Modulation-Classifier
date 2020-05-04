import torch.nn as nn
from functools import partial
from collections import OrderedDict
from pytorch_model_summary import summary
import torch
# from torchsummary import summary


def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


class Conv2dAuto(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.padding = (self.kernel_size[0]//2, self.kernel_size[1]//2)  # adding dynamic padding


conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)

# applies some conv blocks and sum it to input
class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.in_channels,self.out_channels = in_channels,out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()

    def forward(self,x):
        residual =x
        if self.apply_shortcut:  residual=self.shortcut(x)
        x= self.blocks(x)
        x += residual
        return x

    @property
    def apply_shortcut(self):
        return self.in_channels != self.out_channels



# extending the residual block
# identity = conv + activation, referred to as shortcut
# expansion param to increase the out channels
class ResnetResidualBlock(ResidualBlock):
    def __init__(self,in_channels,out_channels,expansion=1,downsampling=1,conv=conv3x3,*args,**kwargs):
        super().__init__(in_channels,out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(OrderedDict(
            {
                'conv': nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                                  stride=self.downsampling, bias=False),
                'bn': nn.BatchNorm2d(self.expanded_channels)
            }
        ))if self.apply_shortcut else None

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def apply_shortcut(self):
        return self.in_channels != self.expanded_channels


# A basic ResNet block is composed by two layers of 3x3 convs/batchnorm/relu
# creating a basic block
def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs),
                          'bn': nn.BatchNorm2d(out_channels)}))


class ResnetBasicBlock(ResnetResidualBlock):
    expansion=1
    def __init__(self,in_channels,out_channels,activation='relu',*args,**kwargs):
        super().__init__(in_channels,out_channels,*args,**kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels,self.out_channels,conv=self.conv,bias=False,stride=self.downsampling),
            activation_func(activation),
            conv_bn(self.out_channels,self.expanded_channels,conv=self.conv,bias=False)
        )


# to increase the network depth while keeping the parameter size small
# introduced bottleneck
# 1x1 to 3x3 to 1x1

class ResnetBottleneckBlock(ResnetResidualBlock):
    expansion=1
    def __init__(self,in_channels,out_channels,activation='relu',*args,**kwargs):
        super().__init__(in_channels,out_channels,expansion=1,*args,**kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, conv=self.conv, kernel_size=1),
            activation_func(activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, kernel_size=3, stride= self.downsampling),
            activation_func(activation),
            conv_bn(self.out_channels, self.expanded_channels, conv=self.conv, kernel_size=1)
        )


# stacking up 'n' resnet blocks
class ResnetLayer(nn.Module):
    def __init__(self,in_channels,out_channels, block=ResnetBasicBlock, n=1,*args,**kwargs):
        super().__init__()
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels,out_channels,*args,**kwargs,downsampling=downsampling),
            *[block(out_channels * block.expansion,
                    out_channels,downsampling=1,*args,**kwargs) for _ in range(n-1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

# dummy = torch.ones((1,64,48,48))
# layer = ResnetLayer(64,128,block=ResnetBasicBlock,n=3)
# print(layer(dummy).shape)

# encoder: conv layers basically
class ResnetEncoder(nn.Module):
    def __init__(self,in_channels=2,blocks_sizes=[64,128,256,512],depths=[2,2,2,2],
                 activation='relu',block=ResnetBasicBlock,*args,**kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation_func(activation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        print(self.in_out_block_sizes)
        self.blocks = nn.ModuleList([
            ResnetLayer(blocks_sizes[0], blocks_sizes[0], n=depths[0], activation=activation,
                        block=block, *args, **kwargs),
            *[ResnetLayer(in_channels * block.expansion,
                          out_channels, n=n, activation=activation,
                          block=block, *args, **kwargs)
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:])]
        ])


    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            # print(block)
            x = block(x)
        return x


# decoder : FC layers after the conv layers
class ResnetDecoder(nn.Module):
    def __init__(self,in_features,n_classes,n_fc_neurons=128):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(in_features,n_fc_neurons)
        self.fc2 = nn.Linear(n_fc_neurons,n_fc_neurons)
        self.fc3 = nn.Linear(n_fc_neurons,n_classes)
        self.activation = activation_func('selu')
        self.dropout = nn.AlphaDropout()

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x


# final model
class Resnet(nn.Module):
    def __init__(self,in_channels,n_classes,*args,**kwargs):
        super().__init__()
        self.encoder = ResnetEncoder(in_channels,*args,**kwargs)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels,n_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(dim=3)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def resnet18(in_channels, n_classes):
    return Resnet(in_channels, n_classes, block=ResnetBasicBlock, depths=[2, 2, 2, 2])

def resnet34(in_channels, n_classes):
    return Resnet(in_channels, n_classes, block=ResnetBasicBlock, depths=[3, 4, 6, 3])

def resnet50(in_channels, n_classes):
    return Resnet(in_channels, n_classes, block=ResnetBottleneckBlock, depths=[3, 4, 6, 3])

def resnet101(in_channels, n_classes):
    return Resnet(in_channels, n_classes, block=ResnetBottleneckBlock, depths=[3, 4, 23, 3])

def resnet152(in_channels, n_classes):
    return Resnet(in_channels, n_classes, block=ResnetBottleneckBlock, depths=[3, 8, 36, 3])


if __name__=="__main__":
    model = resnet50(2,8)
    # print(summary(model.cuda(),(3,224,224)))
    print(summary(model, torch.ones((1, 1024, 2))))