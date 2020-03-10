import hydra
from omegaconf import DictConfig

import torch
import torch.nn as nn

from torchsummary import  summary
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from model import Model

class DenseNet_official(Model):
    def __init__(self,**configs):
        super(Model,self).__init__()
        self.layers=configs["layers"]
        self.epochs=configs["epochs"]
        self.batch_size=configs["batch_size"]
        self.log_dir=configs["log_dir"]
        self.ckpts_dir=configs["ckpts_dir"]
        self.lr=configs["lr"]
        self.momentum=configs["momentum"]
        self.verbose_step=configs["verbose_step"]
        self.verbose=configs["verbose"]
        self.num_classes=configs["num_classes"]
        self.net_type=configs["model"]
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"{self.device}")
        
        self.train_transforms=transforms.Compose([transforms.Resize((224,224),interpolation=2),
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],
                std=[0.5,0.5,0.5])])
        
        self.test_transforms=transforms.Compose([transforms.Resize((224,224),interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],
                std=[0.5,0.5,0.5])])

       
        if self.net_type=="densenet121":
            self.net=models.densenet121(pretrained=False,num_classes=self.num_classes)
        elif self.net_type=="densenet169":
            self.net=models.densenet169(pretrained=False,num_classes=self.num_classes)
        elif self.net_type=="densenet201":
            self.net=models.densenet201(pretrained=False,num_classes=self.num_classes)
        elif self.net_type=="densenet264":
            self.net=models.densenet264(pretrained=False,num_classes=self.num_classes)
        else:
            print(f"Error: {self.net_type} is not recognized")
            return None

        if torch.cuda.device_count()>0:
            self.net=nn.DataParallel(self.net)
            print(f"Number of GPUs {torch.cuda.device_count()}")
        self.net.to(self.device)
        
        if torch.cuda.device_count()<=1 and self.verbose==1:
            summary(self.net,(3,224,224))

@hydra.main("./default.yaml")
def main(cfg):
    configs=cfg["parameters"]
    print(configs)
    model=DenseNet_official(**configs)
    model.train()
    model.test()

    print("Success")

if __name__=="__main__":
    main()   
