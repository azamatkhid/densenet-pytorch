import torch
import torchvision.models as models
from torchsummary import summary

densenet=models.densenet121(pretrained=False)
summary(densenet,(3,224,224))
