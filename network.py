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
        self.conv1=nn.Conv2d(inchannel,4*outchannel,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn2=nn.BatchNorm2d(4*outchannel)
        self.act2=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(4*outchannel,outchannel,kernel_size=3,stride=1,padding=1,bias=False)
        

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
    def __init__(self,inchannel):
        super(TransitionLayer,self).__init__()
        outchannel=inchannel//2
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
    def __init__(self,layers,inchannel,outchannel,growth,num_classes=1000):
        super(DenseNet,self).__init__()
        self.conv0=nn.Conv2d(inchannel,outchannel,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn0=nn.BatchNorm2d(outchannel)
        self.act0=nn.ReLU(inplace=True)
        self.pool0=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1=DenseLayer(outchannel,growth,layers[0])
        self.tr_layer1=TransitionLayer(outchannel+layers[0]*growth)
        outchannel=(outchannel+layers[0]*growth)//2
        self.layer2=DenseLayer(outchannel,growth,layers[1])
        self.tr_layer2=TransitionLayer(outchannel+layers[1]*growth)
        outchannel=(outchannel+layers[1]*growth)//2
        self.layer3=DenseLayer(outchannel,growth,layers[2])
        self.tr_layer3=TransitionLayer(outchannel+layers[2]*growth)
        outchannel=(outchannel+layers[2]*growth)//2
        self.layer4=DenseLayer(outchannel,growth,layers[3])
        self.bn1=nn.BatchNorm2d(outchannel+growth*layers[3])
        self.globpooling=nn.AdaptiveAvgPool2d((1,1))
        self.layer5=nn.Linear(outchannel+growth*layers[3],num_classes)

    def forward(self,x):
        out=self.conv0(x)
        out=self.act0(self.bn0(out))
        out=self.pool0(out)
        out=self.tr_layer1(self.layer1(out))
        out=self.tr_layer2(self.layer2(out))
        out=self.tr_layer3(self.layer3(out))
        out=self.globpooling(self.bn1(self.layer4(out)))
        out=torch.flatten(out,1)
        out=self.layer5(out)
        return out


if __name__=="__main__":
    denseBC=DenseBlock(64,32)
    summary(denseBC,(64,56,56))

    denseLayer=DenseLayer(64,32,6)
    summary(denseLayer,(64,56,56))

    tranLayer=TransitionLayer(256)
    summary(tranLayer,(256,56,56))

    densenet=DenseNet([6,12,24,16],3,64,32)
    summary(densenet,(3,224,224))
