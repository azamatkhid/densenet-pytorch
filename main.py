import os
import hydra
from omegaconf import DictConfig

from app import Application

@hydra.main("./default.yaml")
def main(cfg):
    app=Application(cfg.parameters)
    app.train()
    app.test()

    print("Success")

if __name__=="__main__":
    main()
