import argparse
import os
import random
import shutil
import time
import warnings
import random
from enum import Enum
from torch.nn import functional as F
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter

from cifar_mixup import CIFAR100
from split_val_test import get_train_valid_loader

import logging
logging.basicConfig()
logging.root.setLevel(logging.NOTSET)



best_acc1 = 0



def main(args):
    global best_acc1
    global device

    writer = SummaryWriter(log_dir=os.path.join("tensorboard", args.experiment_name))

    if torch.cuda.is_available():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info("Use GPU for training : CUDA version {}, device:{}".format(torch.version.cuda,device))

    # create model
    teacher=args.teacher
    student=args.student
    logging.info("=> Loading Student model '{}'".format(student._get_name()))
    logging.info("=> Loading Teacher model '{}'".format(teacher._get_name()))
        

    if not torch.cuda.is_available():
        logging.info('using CPU, this will be slow')

    teacher = teacher.to(device)
    student = student.to(device)

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion_val = nn.CrossEntropyLoss().to(device)
    criterion=torch.nn.KLDivLoss(reduction="sum").to(device)

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    """scheduler = StepLR(optimizer, step_size=250, gamma=0.1)"""
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = args.epochs + 1, T_mult=1, eta_min=args.min_lr, verbose=False)
    
    # optionally resume from a checkpoint
    if args.resume:
        
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            if torch.cuda.is_available():
                checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            
            student.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logging.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))


    train_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    val_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


   
    train_dataset = CIFAR100(root='./data', train=True,
                                            download=True, transform=train_transform)
    
    test_loader, val_loader = get_train_valid_loader("./data",
                            batch_size = 4,
                            transform_list=val_transform,
                            random_seed = 1,
                            valid_size=0.5,
                            shuffle=True,
                            show_sample=False,
                            num_workers=1,
                            pin_memory=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)


    """ if args.evaluate:
        validate(val_loader, student, criterion, args)
        return"""

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        writer = train(train_loader, student,teacher, criterion, optimizer, epoch, device, writer, args)

        # evaluate on validation set # change this as this is the measure of whether the model is good or not versus the training/ validation set
        acc1, writer = validate(val_loader, student, criterion_val, writer, epoch, args)
        
        scheduler.step()
        
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)


        save_checkpoint(
                state={'epoch': epoch + 1,
                'student': args.student,
                'state_dict': student.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
                }, 
                is_best=is_best,
                args=args,
                filename=os.path.join("checkpoints",args.experiment_name +'_checkpoint.pth.tar'))
    writer.flush()

def train(train_loader, student,teacher, criterion, optimizer, epoch, device, writer, args):
    batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    top1_teacher = AverageMeter('Teacher Acc@1', ':6.2f')
    top5_teacher = AverageMeter('Teacher Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1, top5, top1_teacher, top5_teacher],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    student.train()
    teacher.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        # data_time.update(time.time() - end)

        # move data to the same device as model
        images = images.to(device)#, non_blocking=True)
        target = target.to(device)#, non_blocking=True)
        # writer.add_graph(student, images)
        # writer.add_graph(teacher, images)
        optimizer.zero_grad()
        
        # compute output
        output_student_raw = student(images)
        output_teacher_raw = teacher(images)



        output_student = F.log_softmax(output_student_raw/args.temperature, dim=1)
        output_teacher = F.softmax(output_teacher_raw/args.temperature, dim=1)
        
        loss = criterion(output_student, output_teacher)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output_student_raw, target, topk=(1, 5))
        #for the teacher
        t_acc1, t_acc5 = accuracy(output_teacher_raw, target, topk=(1, 5))

        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        top1_teacher.update(t_acc1[0], images.size(0))
        top5_teacher.update(t_acc5[0], images.size(0))
        # compute gradient and do SGD step
        
        loss.backward()
        optimizer.step()
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Accuracy_student/train", acc1, epoch)
        writer.add_scalar("Accuracy_teacher/train", t_acc1, epoch)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)
    return writer
    

def validate(val_loader, student, criterion, writer, epoch, args):

    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                # compute output
                output = student(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + ((len(val_loader.sampler) < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    student.eval()

    run_validate(val_loader)
    # if args.distributed:
    #     top1.all_reduce()
    #     top5.all_reduce()

    if (len(val_loader.sampler) < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler), len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()
    
    writer.add_scalar("Loss_student/Val", losses.avg, epoch)
    writer.add_scalar("Accuracy_student/Val", top1.avg, epoch)

    return top1.avg, writer


def save_checkpoint(state, is_best, args,filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join('best_models',args.experiment_name+'_model_best.pth.tar'))

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        # dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        logging.info(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
def add_pr_curve_tensorboard(class_index, test_probs, test_label,classes, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    tensorboard_truth = test_label == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_truth,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()