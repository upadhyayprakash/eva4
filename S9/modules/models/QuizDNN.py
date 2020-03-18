import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.1
class QuizDNN(nn.Module):
    def __init__(self):
        super(QuizDNN, self).__init__()
        
        # CONVOLUTION BLOCK 1
        # self.convblock1 = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(3, 3), padding=0, bias=False),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(128),
        #     nn.Dropout(dropout_value)
        # ) # output_size = 30

        # self.pool1 = nn.MaxPool2d(2, 2) # output_size = 15

        # # CONVOLUTION BLOCK 2
        # self.convblock2 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=0, bias=False, dilation=2),
        #     nn.ReLU(),
        #     nn.BatchNorm2d(256),
        #     nn.Dropout(dropout_value)
        # ) # output_size = 11
        
        # # CONVOLUTION BLOCK 3
        # self.convblock3 = nn.Sequential(
        #     nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1), padding=0, bias=False, groups=256),
        #     # nn.ReLU(),            
        #     # nn.BatchNorm2d(64),
        #     # nn.Dropout(dropout_value)
        # ) # output_size = 5

        # self.pointwise = nn.Sequential(
        #     nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), padding=0, bias=False),
        #     # nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), padding=0, bias=False),
        #     nn.ReLU(),            
        #     nn.BatchNorm2d(128),
        #     nn.Dropout(dropout_value)
        # ) # output_size = 5

        # # GAP Layer
        # self.gap = nn.Sequential(
        #     nn.AvgPool2d(kernel_size=2)
        # ) # output_size = 3

        # # CONVOLUTION BLOCK 4
        # self.convblock4 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), padding=0, bias=False, stride=1),
        #     nn.ReLU(),            
        #     nn.BatchNorm2d(64),
        #     nn.Dropout(dropout_value)
        # ) # output_size = 1

        # self.convblock5 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), padding=0, bias=False, stride=1),
        #     nn.ReLU(),            
        #     nn.BatchNorm2d(64),
        #     nn.Dropout(dropout_value)
        # ) # output_size = 1

        # self.convblock6 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), padding=0, bias=False, stride=1),
        #     nn.ReLU(),            
        #     nn.BatchNorm2d(256),
        #     nn.Dropout(dropout_value)
        # ) # output_size = 1

        # self.convblock7 = nn.Sequential(
        #     nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), padding=0, bias=False, stride=1),
        #     nn.ReLU(),            
        #     nn.BatchNorm2d(128),
        #     nn.Dropout(dropout_value)
        # ) # output_size = 1

       	# self.convblock8 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), padding=0, bias=False, stride=1),
        #     nn.ReLU(),            
        #     nn.BatchNorm2d(64),
        #     nn.Dropout(dropout_value)
        # ) # output_size = 1


        # self.FC = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        #     # nn.BatchNorm2d(10),
        #     # nn.ReLU(),
        #     # nn.Dropout(dropout_value)
        # )


        self.conv1 = nn.Conv2d(3, 128, 3, padding=1) #input -? OUtput? RF
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 32, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv7 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv8 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv9 = nn.Conv2d(128, 32, 3, padding=1)
        
        self.gap = nn.AvgPool2d(kernel_size=2)
        self.fc = nn.Linear(-1, 10)


    def forward(self, x):

        # === OLD Version
        
        # x = self.convblock1(x)
        # x = self.pool1(x)
        # x = self.convblock2(x)
        # x = self.pool1(x)
        # x = self.convblock3(x)
        # x = self.pointwise(x)
        # x = self.gap(x)
        # x = self.convblock4(x)
        # x = self.convblock5(x)
        # x = x.view(-1, 10)
        
        # return x

        # === NEW Version
		x1 = x
		x2 = F.relu(self.conv1(x1))
		print(x2.shape)
		x3 = self.conv1(torch.cat((x1+x2), dim=1))
		print(x2.shape)
		x4 = self.pool1(x3)
		x5 = self.convblock3(x4)
		print(x2.shape)
		x6 = self.convblock4(x5)
		print(x2.shape)
		x7 = self.convblock5(x6)
		x8 = self.pool1(x7)
		x9 = self.convblock6(x8)
		x10 = self.convblock7(x9)
		x11 = self.convblock8(x10)
		x12 = self.gap(x11)
		x13 = self.FC(x12)
		xOut = x13.view(-1, 10)