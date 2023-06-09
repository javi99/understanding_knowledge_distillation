import torch
from main_training import main
from torchvision.models import mobilenet_v3_small

class argclass(object):
    def __init__(self, *initial_data,**kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])



######## for experiment 1 ########
#Considering different teachers :  

#teacher1 = torch.hub.load("chenyaofo/pytorch-cifar-models", 'cifar100_resnet44', pretrained=True)
#teacher2= torch.hub.load("chenyaofo/pytorch-cifar-models", 'cifar100_vgg11_bn', pretrained=True)
teacher3=torch.hub.load("chenyaofo/pytorch-cifar-models",'cifar100_mobilenetv2_x0_75',pretrained=True)


hyperparams1={'lr':0.003,
              'min_lr':0,
             'momentum':0.9,
             'weight_decay':1e-4,
             'batch_size':512,
             'epochs':300,
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
             "experiment_name":"test2-teacher3-T5-lr0003",
             'temperature':5
            }
hyperparams2={'lr':0.001,
              'min_lr':0,
             'momentum':0.9,
             'weight_decay':1e-4,
             'batch_size':512,
             'epochs':300,
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
             "experiment_name":"test2-teacher3-T10-lr0001",
             'temperature':10
            }
hyperparams3={'lr':0.001,
              'min_lr':0,
             'momentum':0.9,
             'weight_decay':1e-4,
             'batch_size':512,
             'epochs':300,
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
             "experiment_name":"test2-teacher3-T2-lr0001",
             'temperature':2
            }

hyperparams4={'lr':0.01,
              'min_lr':0,
             'momentum':0.9,
             'weight_decay':1e-4,
             'batch_size':512,
             'epochs':300,
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
             "experiment_name":"test2-teacher3-T2-lr001",
             'temperature':2
            }

args1=argclass(hyperparams1)
args2=argclass(hyperparams2)
args3=argclass(hyperparams3)
args4=argclass(hyperparams4)

if __name__ == "__main__":
    main(args=args1)
    main(args=args2)
    main(args=args3)
    main(args=args4)