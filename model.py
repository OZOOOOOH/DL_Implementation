import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")


class Net(nn.Module):
    def __init__(self,num_classes=2,init_weights=True):
        super().__init__()

        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU()
        self.pool1=nn.MaxPool2d(3,stride=2,padding=1)

        self.layer1 = self.make_layer(64, 64, 3, stride=1)
        self.layer2 = self.make_layer(256, 128, 4, stride=2)
        self.layer3 = self.make_layer(512, 256, 6, stride=2)
        self.layer4 = self.make_layer(1024, 512, 3, stride=2)


        self.AvgPool=nn.AdaptiveAvgPool2d((1,1))

        self.fc1=nn.Linear(512*4,num_classes)

        # weights inittialization
        if init_weights:
            self._initialize_weights()


    def make_layer(self,in_channels, out_channels, num_blocks, stride=1):

        strides = [stride] + [1] * (num_blocks - 1)

        layers = []

        for stride in strides:

            layers.append(Block(in_channels, out_channels, stride))

            in_channels = out_channels * 4

        return nn.Sequential(*layers)



    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.pool1(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)


        x=self.AvgPool(x)

        x=x.view(x.size(0),-1)

        x=F.relu(self.fc1(x))
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
class Block(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride,bias=False),

            nn.BatchNorm2d(out_dim),

            nn.ReLU(),

            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1,bias=False),

            nn.BatchNorm2d(out_dim),

            nn.ReLU(),

            nn.Conv2d(out_dim, out_dim * 4, kernel_size=1,bias=False),

            nn.BatchNorm2d(out_dim * 4),

        )
        self.shortcut=nn.Sequential()

        self.relu = nn.ReLU()

        if stride!=1 or in_dim!=out_dim*4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, out_dim * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_dim * 4),
            )

    def forward(self, x):
        f = self.block(x)
        x=self.shortcut(x)
        h = f + x
        h = self.relu(h)

        return h