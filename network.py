import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary

class DenseBlock(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(DenseBlock,self).__init__()
        self.bn1=nn.BatchNorm2d(inchannel)
        self.act1=nn.ReLU(inplace=True)
        self.conv1=nn.Conv2d(inchannel,outchannel,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn2=nn.BatchNorm2d(outchannel)
        self.act2=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(outchannel,outchannel,kernel_size=3,stride=1,padding=1,bias=False)
        

    def forward(self,x):
        out=self.bn1(x)
        out=self.act1(out)
        out=self.conv1(out)
        out=self.bn2(out)
        out=self.act2(out)
        out=self.conv2(out)
        return torch.cat((x,out),dim=1)

class DenseLayer(nn.Module):
    def __init__(self,inchannel,growth,num_blocks):
        super(DenseLayer,self).__init__()
        self.inchannel=inchannel
        self.k=growth
        self.num_blocks=num_blocks
        layers=[]
        inn,out=inchannel,growth
        for i in range(self.num_blocks):
            layers.append(DenseBlock(inn,out))
            inn+=out
        self.layer=nn.Sequential(*layers)


    def forward(self,x):
        return self.layer(x)

class TransitionLayer(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(TransitionLayer,self).__init__()
        self.bn1=nn.BatchNorm2d(inchannel)
        self.act1=nn.ReLU(inplace=True)
        self.conv1=nn.Conv2d(inchannel,outchannel,kernel_size=1,stride=1,padding=0,bias=False)
        self.pooling=nn.AvgPool2d(kernel_size=2,stride=2,padding=0)

    def forward(self,x):
        out=self.bn1(x)
        out=self.act1(out)
        out=self.conv1(out)
        out=self.pooling(out)
        return out

class DenseNet(nn.Module):
    def __init__(self,layers):

    def _make_dense_layer(self):

    def forward(self,x)


if __name__=="__main__":
    denseBC=DenseBlock(64,32)
    summary(denseBC,(64,56,56))

    denseLayer=DenseLayer(64,32,6)
    summary(denseLayer,(64,56,56))

    tranLayer=TransitionLayer(256,64)
    summary(tranLayer,(256,56,56))
