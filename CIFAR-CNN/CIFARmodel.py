from torch import flatten
import torch.nn as nn

class Net(nn.Module):
    def __init__(self,Channels,classes):
        super().__init__()
        #conv->relu->pool
        self.conv1 = nn.Conv2d(Channels,16,3,padding=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2,2)
        #conv->relu->pool
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2,2)
        #conv->relu->pool
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(2,2)
        #FC layer 1
        self.fc1 = nn.Linear(in_features=64*4*4,out_features=512)
        self.relu4 = nn.ReLU()
        #FC layer 2
        self.fc2 = nn.Linear(512,256)
        self.relu5 = nn.ReLU()
        #FC layer 3
        self.fc3 = nn.Linear(256,64)
        self.relu6 = nn.ReLU()
        #Output 
        self.fc4 = nn.Linear(64,classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)
    def forward(self,x):
        #layer 1
        x=self.conv1(x)
        x=self.relu1(x)
        x=self.maxpool1(x)
        #layer 2
        x=self.conv2(x)
        x=self.relu2(x)
        x=self.maxpool2(x)
        #layer 3
        x=self.conv3(x)
        x=self.relu3(x)
        x=self.maxpool3(x)
        #layer 4
        x=flatten(x,1)
        x=self.fc1(x)
        x=self.relu4(x)
        #layer 5
        x=self.fc2(x)
        x=self.relu5(x)
        #layer 6
        x=self.fc3(x)
        x=self.relu6(x)
        #output layer
        x=self.fc4(x)
        x=self.logsoftmax(x)
        return x