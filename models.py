## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the  import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # Output size = (224- 5)/1 + 1 = 220
        # output tensor dimensions: (32 , 220, 220)
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        # output tensor dimensions: (32, 110, 110)
        self.pool1= nn.MaxPool2d(2, 2)
        
        # dropout with p=0.1
        self.fc1_drop = nn.Dropout(p=0.1)
        
        # Output size = (110-4)/1 + 1 = 107
        # output tensor dimensions: (64 , 110, 110)
        self.conv2 = nn.Conv2d(32, 64, 4)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        # output tensor dimensions: (64, 53, 53)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # dropout with p=0.2
        self.fc2_drop = nn.Dropout(p=0.2)
        
        # Output size = (53-3)/1 + 1 = 51
        # output tensor dimensions: (128, 51, 51)
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        # output tensor dimensions: (128, 25, 25)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # dropout with p=0.2
        self.fc3_drop = nn.Dropout(p=0.3)
        
        # Output size = (25- 2)/1 + 1 = 24
        # output tensor dimensions: (256 , 24, 24)
        self.conv4 = nn.Conv2d(128, 256, 2)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        # output tensor dimensions: (256, 12, 12)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # dropout with p=0.2
        self.fc4_drop = nn.Dropout(p=0.4)
        
        # Output size = (12 - 1)/1 + 1 = 12
        # output tensor dimensions: (256 , 12, 12)
        self.conv5 = nn.Conv2d(256, 512, 1)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        # output tensor dimensions: (512, 6, 6)
        self.pool5 = nn.MaxPool2d(2, 2)
        
        # dropout with p=0.2
        self.fc5_drop = nn.Dropout(p=0.5)
        
        # 512 outputs * the 6*6 filtered/pooled map size
        self.fc1 = nn.Linear(512*6*6, 1000)
        
        # dropout with p=0.2
        self.fc6_drop = nn.Dropout(p=0.6)
        
        # 1000
        self.fc2 = nn.Linear(1000, 1000)
        
        # dropout with p=0.2
        self.fc7_drop = nn.Dropout(p=0.7)
        
         # 1000
        self.fc3 = nn.Linear(1000, 136)
        

        
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool1(F.elu(self.conv1(x)))
        x = self.fc1_drop(x)   
        
        x = self.pool2(F.elu(self.conv2(x)))
        x = self.fc2_drop(x)
        
        x = self.pool3(F.elu(self.conv3(x)))
        x = self.fc3_drop(x)
        
        x = self.pool4(F.elu(self.conv4(x)))
        x = self.fc4_drop(x)
        
        x = self.pool5(F.elu(self.conv5(x)))
        x = self.fc5_drop(x)
        
        x = x.view(x.size(0), -1)
        x = F.tanh(self.fc1(x))
        x = self.fc6_drop(x)
        
        x = F.tanh(self.fc2(x))
        x = self.fc7_drop(x)
        
        
        x = self.fc3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
