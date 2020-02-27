import os
import hydra
from omegaconf import DictConfig

from model import Model

@hydra.main("./default.yaml")
def main(cfg):
    configs=cfg["parameters"]

    model=Model(**configs)
    model.train()
    model.test()

    print("Success")

if __name__=="__main__":
    main()
