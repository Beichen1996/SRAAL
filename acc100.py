import os
import random
# Torch
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Torchvison
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import CIFAR100, CIFAR10

# Utils
from tqdm import tqdm
import models.resnet as resnet
from config import *
# Data
train_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(size=32, padding=4),
    T.ToTensor(),
    T.Normalize([0.50707543, 0.48655024, 0.44091907], [0.26733398, 0.25643876, 0.27615029]) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
])

test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.50707543, 0.48655024, 0.44091907], [0.26733398, 0.25643876, 0.27615029]) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
])

cifar100_train = CIFAR100('./data', train=True, download=True, transform=train_transform)
cifar100_unlabeled   = CIFAR100('./data', train=True, download=True, transform=test_transform)
cifar100_test  = CIFAR100('./data', train=False, download=True, transform=test_transform)

def train_epoch(models, criterion, optimizers, dataloaders, epoch):
    models.train()
    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        inputs = data[0].cuda()
        labels = data[1].cuda()

        optimizers['backbone'].zero_grad()
       
        scores, features = models(inputs)
        target_loss = criterion(scores, labels)
        m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        loss            = m_backbone_loss 

        loss.backward()
        optimizers['backbone'].step()
    return m_backbone_loss

#
def test(models, dataloaders, mode='val'):
    assert mode == 'val' or mode == 'test'
    models.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores, _ = models(inputs)
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    return 100 * correct / total

#
def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs):
    bestacc = 0.
    for epoch in range(num_epochs):
        schedulers['backbone'].step()
        losss = train_epoch(models, criterion, optimizers, dataloaders, epoch)
        if epoch % 30 == 0:
            acc = test(models, dataloaders, 'test')
            if bestacc < acc:
                bestacc = acc
            if epoch % 30 == 0:
                print('Val Acc: {:.3f}% \t '.format(acc))
    return acc


if __name__ == '__main__':
    vis = None
    plot_data = {'X': [], 'Y': [], 'legend': ['Backbone Loss']}
    EPOCH = 360
    BATCH = 128
    
    torch.backends.cudnn.benchmark = True
    for trial in range(1,7):
        resnet18    = resnet.ResNet18(num_classes=100).cuda()
        models      = resnet18
        name =  './results_cifar100/' + str(int(trial*5+10)) +'.npy'
        print (name)
        indices = np.load( name).tolist()
        labeled_set = indices
        all_indices = set(np.arange(NUM_TRAIN))
        indices = list(range(NUM_TRAIN))
        unlabeled_set = np.setdiff1d(indices, labeled_set).tolist()
        
        train_loader = DataLoader(cifar100_train, batch_size=BATCH, 
                                  sampler=SubsetRandomSampler(labeled_set), 
                                  pin_memory=True)
        test_loader  = DataLoader(cifar100_test, batch_size=BATCH)
        dataloaders  = {'train': train_loader, 'test': test_loader}
        criterion      = nn.CrossEntropyLoss(reduction='none')
        optim_backbone = optim.SGD(models.parameters(), lr=LR, 
                                momentum=MOMENTUM, weight_decay=WDECAY)
        
        sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)

        optimizers = {'backbone': optim_backbone}
        schedulers = {'backbone': sched_backbone}

        # Training and test
        acc = train(models, criterion, optimizers, schedulers, dataloaders, EPOCH)
        print('Label set size {}%: Test acc {}'.format(len(labeled_set)/NUM_TRAIN*100, acc))
