import torch
# from main_training import main
from torchvision.models import mobilenet_v3_small
student = mobilenet_v3_small(pretrained=False, num_classes=100)
teacher1 = torch.hub.load("chenyaofo/pytorch-cifar-models", 'cifar100_resnet44', pretrained=True)
teacher2= torch.hub.load("chenyaofo/pytorch-cifar-models", 'cifar100_vgg11_bn', pretrained=True)
teacher3=torch.hub.load("chenyaofo/pytorch-cifar-models",'cifar100_mobilenetv2_x0_75',pretrained=True)