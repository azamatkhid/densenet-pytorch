import hydra
from omegaconf import DictConfig

import torch
import torch.nn as nn

from torchsummary import  summary
import torchvision.models as models

from app import Application

class DenseNet_official(Application):
    def __init__(self,cfg):
        super(DenseNet_official,self).__init__(cfg)
        pass

    def build(self):
        model=getattr(models,self.cfg.model)
        self.net=model(pretrained=False,num_classes=self.cfg.num_classes)

        if torch.cuda.device_count()>0:
            self.net=nn.DataParallel(self.net)
            print(f"Number of GPUs {torch.cuda.device_count()}")
        
        self.net.to(self.device)

        if torch.cuda.device_count()<=1 and self.cfg.verbose:
            summary(self.net,(3,224,224))

@hydra.main("./default.yaml")
def main(cfg: DictConfig) -> None:
    app=DenseNet_official(cfg.parameters)
    app.build()
    app.train()
    app.test()

    print("Success")

if __name__=="__main__":
    main()   
