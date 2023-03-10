from torch.nn import Module,Conv2d,Linear,MaxPool2d,ReLU,LogSoftmax
from torch import flatten

class LeNet(Module):
    def __init__(self,numChannels,classes):
        super(LeNet,self).__init__()
        #first Conv->relu->pool
        self.conv1 = Conv2d(in_channels=numChannels,out_channels=20,kernel_size=(5,5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2,2),stride=(2,2))
        #second Conv->relu->pool
        self.conv2 = Conv2d(in_channels=20,out_channels=50,kernel_size=(5,5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2,2),stride=(2,2))
        #FC layer
        self.fc1 = Linear(in_features=800, out_features=500)
        self.relu3 = ReLU()
        #softmax output
        self.fc2 = Linear(in_features=500,out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)
    def forward(self,x):
        #layer 1
        x=self.conv1(x)
        x=self.relu1(x)
        x=self.maxpool1(x)
        #layer 2
        x=self.conv2(x)
        x=self.relu2(x)
        x=self.maxpool2(x)
        #flatten and fc layer
        x=flatten(x,1)
        x=self.fc1(x)
        x=self.relu3(x)
        #softmax layer
        x=self.fc2(x)
        output = self.logSoftmax(x)
        return output
    
