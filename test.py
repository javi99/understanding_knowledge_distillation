import torch
from main_training import main
from torchvision.models import mobilenet_v3_small
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt

class argclass(object):
    def __init__(self, *initial_data,**kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

teacher3=torch.hub.load("chenyaofo/pytorch-cifar-models",'cifar100_mobilenetv2_x1_0',pretrained=True)

hyperparams3={'lr':0.001,
              'min_lr':0,
             'momentum':0.9,
             'weight_decay':1e-4,
             'batch_size':512,
             'epochs':400,
             'start_epoch':0,
             'print_freq':50,
             'pretrained':True,
             'evaluate':False,
             'world_size':-1,
             'rank':-1,
             'dist_url':'tcp://',
             'dist_backend':'nccl',
             'seed':None,
             'gpu':None,
             'multiprocessing_distributed':False,
             'dummy':False,
             'data':'./data',
             'workers':4,
            #  'resume':'model_best.pth.tar',
             'resume':'',
             'evaluate':False,
             'teacher':teacher3,
             'student':mobilenet_v3_small(pretrained=False, num_classes=100),
             "experiment_name":"test1-teacher3-lr-cossine",
             'temperature':5
            }

args1=argclass(hyperparams3)

student = args1.student
optimizer = torch.optim.Adam(student.parameters(), lr=args1.lr,
                                weight_decay=args1.weight_decay)

"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = args1.epochs + 1, T_mult=1, eta_min=args1.min_lr, verbose=False)

lr_list = []

for i in range(500):
    lr_list.append(scheduler.get_last_lr())
    scheduler.step()

plt.plot(lr_list)
plt.grid()
plt.xlabel("epoch")
plt.ylabel("lr")
plt.savefig("cos_lr")