import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class BasicBlockMaxPool(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, res_block=None, stride=1):
        super(BasicBlockMaxPool, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        # Newly added codes
        self.layer = self._make_layer_customized(in_planes, planes)

        self.res_block = None

        if not res_block is None:
            self.res_block = nn.Sequential(
                res_block(planes, planes)
            )

    # Newly added Codes
    def _make_layer_customized(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.layer(x)
        if not self.res_block is None:
            x = x + self.res_block(x)
        return x
      
class ResNet(nn.Module):
    def __init__(self, custom_block, res_block, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.linear = nn.Linear(512*custom_block.expansion, num_classes)

        # Newly added Codes
        
        # Preparation to Feed to ResBlock
        self.prep_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # ResNet Block Creation Customized Layers
        self.create_blocks = nn.Sequential(
            custom_block(64, 128, res_block=res_block),
            custom_block(128, 256),
            custom_block(256, 512, res_block=res_block)
        )


        self.max_pool4 = nn.MaxPool2d(4, 4)

    
    def forward(self, x):
        x = self.prep_layer(x)
        x = self.create_blocks(x)
        x = self.max_pool4(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

def ResNet18MaxPool():
    return ResNet(BasicBlockMaxPool, BasicBlock)

def test():
    net = ResNet18MaxPool()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

test()