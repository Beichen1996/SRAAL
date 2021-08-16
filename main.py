import torch
from torchvision import datasets, transforms
import torch.utils.data.sampler  as sampler
import torch.utils.data as data

import numpy as np
import argparse
import random
import os

from custom_datasets import *
import model
from solver import Solver
from utils import *
import arguments
import resnet


def cifar_transformer():
    return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
def save_sample(listt, path):
    np.save(path, np.array(listt))


def main(args):
    if args.dataset == 'cifar10':
        test_dataloader = data.DataLoader(
                datasets.CIFAR10(args.data_path, download=True, transform=cifar_transformer(), train=False),
            batch_size=args.batch_size, drop_last=False)

        train_dataset = CIFAR10(args.data_path)
        args.num_images = 50000
        args.budget = 2500
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        test_dataloader = data.DataLoader(
                datasets.CIFAR100(args.data_path, download=True, transform=cifar_transformer(), train=False),
             batch_size=args.batch_size, drop_last=False)

        train_dataset = CIFAR100(args.data_path)
        args.num_images = 50000
        args.budget = 2500
        args.num_classes = 100
    else:
        raise NotImplementedError

    all_indices = set(np.arange(args.num_images))
    initial_indices = np.load( './results_cifar100/10.npy' ).tolist()
    sampler = data.sampler.SubsetRandomSampler(initial_indices)
    querry_dataloader = data.DataLoader(train_dataset, sampler=sampler, 
            batch_size=args.batch_size, drop_last=True)
            
    args.cuda =  torch.cuda.is_available()
    print('cuda:', args.cuda)
    solver = Solver(args, test_dataloader)
    splits = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    current_indices = list(initial_indices)
    accuracies = []
    for split in splits:
        task_model = resnet.ResNet18(num_classes=args.classes).cuda()
        vae = model.VAE(args.latent_dim , num_classes = int(args.dataset[5:])).cuda()
        discriminator = model.Discriminator(args.latent_dim).cuda()
        unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
        unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
        unlabeled_dataloader = data.DataLoader(train_dataset, 
                sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False)

        # train the models on the current data
        vae, discriminator = solver.train(querry_dataloader,
                                               task_model, 
                                               vae, 
                                               discriminator,
                                               unlabeled_dataloader
                                               )

        unlabeled_dataloader = data.DataLoader(train_dataset, 
                sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False)
        #sample based on discriminator's predicted state
        sampled_indices = solver.sample_for_labeling(vae, discriminator, unlabeled_dataloader)
        current_indices = list(current_indices) + list(sampled_indices)
        # save the selection into .npy file
        save_sample(current_indices, './results_cifar100/'+str(int(split*100))+'.npy') 
        print(str(int(split*100))+'% samples is selected' )
        sampler = data.sampler.SubsetRandomSampler(current_indices)
        querry_dataloader = data.DataLoader(train_dataset, sampler=sampler, 
                batch_size=args.batch_size, drop_last=True)

 
if __name__ == '__main__':
    args = arguments.get_args()
    main(args)