import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary

#class DenseBlock(nn.Module):
#    def __init__(self,inchannel,outchannel):
#        super(DenseBlock,self).__init__()
#        self.bn1=nn.BatchNorm2d(inchannel)
#        self.act1=nn.ReLU(inplace=True)
#        self.conv1=nn.Conv2d(inchannel,4*outchannel,kernel_size=1,stride=1,padding=0,bias=False)
#        self.bn2=nn.BatchNorm2d(4*outchannel)
#        self.act2=nn.ReLU(inplace=True)
#        self.conv2=nn.Conv2d(4*outchannel,outchannel,kernel_size=3,stride=1,padding=1,bias=False)
#        
#
#    def forward(self,x):
#        out=self.bn1(x)
#        out=self.act1(out)
#        out=self.conv1(out)
#        out=self.bn2(out)
#        out=self.act2(out)
#        out=self.conv2(out)
#        return torch.cat((x,out),dim=1)
#
#class TransitionLayer(nn.Module):
#    def __init__(self,inchannel):
#        super(TransitionLayer,self).__init__()
#        outchannel=inchannel//2
#        self.bn1=nn.BatchNorm2d(inchannel)
#        self.act1=nn.ReLU(inplace=True)
#        self.conv1=nn.Conv2d(inchannel,outchannel,kernel_size=1,stride=1,padding=0,bias=False)
#        self.pooling=nn.AvgPool2d(kernel_size=2,stride=2,padding=0)
#
#    def forward(self,x):
#        out=self.bn1(x)
#        out=self.act1(out)
#        out=self.conv1(out)
#        out=self.pooling(out)
#        return out


class _DenseBlock(nn.Module):
    def __init__(self,inchannel,growth,num_blocks):
        super(_DenseBlock,self).__init__()
        self.inchannel=inchannel
        self.k=growth
        self.num_blocks=num_blocks
        layers=[]
        inn,out=inchannel,growth
        for i in range(self.num_blocks):
            layers.append(self._make_dense_block(inn,out))
            inn+=out
        self.layers=nn.ModuleList(layers)
    
    def _make_dense_block(self,inchannel,outchannel):
        return nn.Sequential(nn.BatchNorm2d(inchannel),
                nn.ReLU(inplace=True),
                nn.Conv2d(inchannel,4*outchannel,kernel_size=1,stride=1,padding=0,bias=False),
                nn.BatchNorm2d(4*outchannel),
                nn.ReLU(inplace=True),
                nn.Conv2d(4*outchannel,outchannel,kernel_size=3,stride=1,padding=1,bias=False))

    def forward(self,x):
        inp=x
        for layer in self.layers:
            out=layer(inp)
            inp=torch.cat((out,inp),dim=1)
        return inp

class Model(nn.Module):
    def __init__(self,layers,inchannel,outchannel,growth,num_classes=1000):
        super(Model,self).__init__()
        self._layers=list(layers)
        self.conv0=nn.Conv2d(inchannel,outchannel,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn0=nn.BatchNorm2d(outchannel)
        self.act0=nn.ReLU(inplace=True)
        self.pool0=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        
        _blocks=[]
        
        for num_blocks in self._layers[:-1]:
            _blocks.append(_DenseBlock(outchannel,growth,num_blocks))
            _blocks.append(self._make_transition(outchannel+num_blocks*growth))
            outchannel=(outchannel+num_blocks*growth)//2

        _blocks.append(_DenseBlock(outchannel,growth,self._layers[-1]))
        _blocks.append(nn.BatchNorm2d(outchannel+growth*self._layers[-1]))

        self.blocks=nn.ModuleList(_blocks)
        self.lastlayer=nn.Linear(outchannel+growth*self._layers[-1],num_classes)
   
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)


    def _make_transition(self,inchannel):
        outchannel=inchannel//2
        return nn.Sequential(nn.BatchNorm2d(inchannel), nn.ReLU(inplace=True),
                nn.Conv2d(inchannel,outchannel,kernel_size=1,stride=1,padding=0,bias=False),
                nn.AvgPool2d(kernel_size=2,stride=2,padding=0))

    def forward(self,x):
        out=self.conv0(x)
        out=self.act0(self.bn0(out))
        out=self.pool0(out)
        
        for block in self.blocks:
            out=block(out)
        out=nn.functional.relu(out,inplace=True)
        out=nn.functional.adaptive_avg_pool2d(out,(1,1))
        out=torch.flatten(out,1)
        out=self.lastlayer(out)
        return out


if __name__=="__main__":
#    denseBC=DenseBlock(64,32)
#    summary(denseBC,(64,56,56))
#
#    denseLayer=_DenseBlock(64,32,6)
#    summary(denseLayer,(64,56,56))
#
#    tranLayer=TransitionLayer(256)
#    summary(tranLayer,(256,56,56))

    densenet=Model([6,12,24,16],3,64,32)
    summary(densenet,(3,224,224))
