import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")



class BottleneckBlock(nn.Module):# bn->relu->1x1 conv->bn->relu->3x3 conv -> concat
    def __init__(self,in_channels,growth_rate):
        super().__init__()
        self.bn1=nn.BatchNorm2d(in_channels)
        self.relu=nn.ReLU()
        self.conv1=nn.Conv2d(in_channels,growth_rate*4,kernel_size=1,bias=False)
        self.bn2=nn.BatchNorm2d(growth_rate*4)
        self.conv2=nn.Conv2d(growth_rate*4,growth_rate,kernel_size=3,padding=1,bias=False)

        self.out_dim = in_channels + self.conv2.out_channels

    def forward(self,x):
        out=self.conv1(self.relu(self.bn1(x)))
        out=self.conv2(self.relu(self.bn2(out)))
        out=torch.cat([x,out],1)
        return out
class TransitionBlock(nn.Module):# bn->relu->1x1 conv->avgPool
    def __init__(self,in_channels,compress=0.5):
        super().__init__()
        self.bn=nn.BatchNorm2d(in_channels)
        self.relu=nn.ReLU()
        self.conv=nn.Conv2d(in_channels=in_channels,out_channels=int(in_channels*compress),kernel_size=1,stride=1,bias=False)
        self.avgPool=nn.AvgPool2d(2,stride=2)

    def forward(self,x):
        out=self.conv(self.relu(self.bn(x)))
        out=self.avgPool(out)
        return out
class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate):
        super(DenseBlock, self).__init__()
        self.convblocks = nn.ModuleList()
        for i in range(num_layers):
            block = BottleneckBlock(in_channels, growth_rate)
            in_channels = block.out_dim
            self.convblocks.append(block)

    def forward(self, x):
        # x = self.convblock(x)
        for layer in self.convblocks:
            x = layer(x)
        return x
class Densenet(nn.Module):
    def __init__(self,in_channels,dense_layers,growth_rate,num_classes=2,compress=0.5):
        super().__init__()
        self.conv=nn.Conv2d(
            in_channels=in_channels,
            out_channels=growth_rate*2,
            kernel_size=7,
            stride=2,
            padding=3
        )
        self.bn=nn.BatchNorm2d(self.conv.out_channels)
        self.pool=nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.relu=nn.ReLU()

        self.denseblocks=nn.ModuleList()
        num_channels=self.conv.out_channels
        len_DenseBlocks = len(dense_layers)


        for i, num_layers in enumerate(dense_layers):
            self.denseblocks.append(
                DenseBlock(num_channels, num_layers, growth_rate)
            )
            num_channels += num_layers * growth_rate
            if i != len_DenseBlocks:
                self.denseblocks.append(
                    TransitionBlock(in_channels=num_channels, compress=compress)
                )
                num_channels = int(num_channels * compress)
        self.fc = nn.Linear(num_channels, num_classes)

        for m in self.modules():#initialize
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x=self.pool(self.relu(self.bn(self.conv(x))))
        for block in self.denseblocks:
            x=block(x)
        x=F.adaptive_avg_pool2d(x,(1,1))
        x=torch.flatten(x,1)
        x=self.fc(x)
        return x

def _test():
    net = Densenet(in_channels=3, dense_layers=[6, 12, 24, 16], growth_rate=32, num_classes=2, compress=0.5).cuda()
    out = net(torch.randn(8, 3, 224, 224).cuda())
    print(out.size())
    # summary(model, (3, 224, 224))
_test()