import os
import hydra
from omegaconf import DictConfig

from app import Application

@hydra.main("./default.yaml")
def main(cfg):
    configs=cfg["parameters"]

    app=Application(**configs)
    app.train()
    app.test()

    print("Success")

if __name__=="__main__":
    main()
