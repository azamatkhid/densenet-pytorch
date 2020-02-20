import torch
import torchvision.models as models

def main():
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net=models.densenet121(pretrained=False)
    print(net)

if __name__=="__main__":
    main()
