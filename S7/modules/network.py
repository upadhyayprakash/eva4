import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.1
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        ) # output_size = 30

        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 15

        # CONVOLUTION BLOCK 2
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=0, bias=False, dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(dropout_value)
        ) # output_size = 11
        
        # CONVOLUTION BLOCK 3
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), padding=0, bias=False, groups=256),
            # nn.ReLU(),            
            # nn.BatchNorm2d(64),
            # nn.Dropout(dropout_value)
        ) # output_size = 5

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), padding=0, bias=False),
            # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        ) # output_size = 5

        # GAP Layer
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=2)
        ) # output_size = 3

        # CONVOLUTION BLOCK 4
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), padding=0, bias=False, stride=2),
            nn.ReLU(),            
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        ) # output_size = 1

       
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        )


    def forward(self, x):
        x = self.convblock1(x)
        x = self.pool1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.pointwise(x)
        x = self.gap(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = x.view(-1, 10)
        
        return x