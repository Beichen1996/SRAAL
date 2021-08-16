import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import resnet

import os
import numpy as np
from sklearn.metrics import accuracy_score

import sampler

import sys



class Solver:
    def __init__(self, args, test_dataloader):
        self.args = args
        self.test_dataloader = test_dataloader
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.sampler = sampler.AdversarySampler(self.args.budget)

    def read_data(self, dataloader, labels=True):
        if labels:
            while True:
                for img, label, indices in dataloader:
                    yield img, label
        else:
            while True:
                for img, _, indices in dataloader:
                    yield img, indices

    def train(self, querry_dataloader, task_model, vae, discriminator, unlabeled_dataloader):
        labeled_data = self.read_data(querry_dataloader)
        unlabeled_data = self.read_data(unlabeled_dataloader, labels=False)

        optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)

        vae.train()
        discriminator.train()
        task_model.train()
        change_lr_iter = self.args.train_iterations // 30

        # task_model for online uncertainty indicator
        print('Train the online uncertainty indicator:')
        criterion      = nn.CrossEntropyLoss(reduction='none')
        optim_backbone = optim.SGD(task_model.parameters(), lr=0.1, 
                                    momentum=0.9, weight_decay=5e-4)
        sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones= [200])
        optimizers = optim_backbone
        schedulers = sched_backbone
        resnet.train(task_model, criterion, optimizers, schedulers, querry_dataloader, self.args.task_epochs)
        scoredic = resnet.uncertainty_score(task_model, unlabeled_dataloader, self.args.classes)
        #Above dict contains all the uncertainty scores for unlabeled samples
        #They will relabel all the unlabeled data's state. ps: the original state for unlabeled data is 1.

        print('The online uncertainty indicator is ready.')
        print('Train the generator and the discriminator.')
        #Begin to train the generator (VAE) and the discriminator.
        for iter_count in range(self.args.train_iterations):
            labeled_imgs, labels = next(labeled_data)
            unlabeled_imgs, indices  = next(unlabeled_data)
            indices = indices.cpu().numpy()
            if self.args.cuda:
                labeled_imgs = labeled_imgs.cuda()
                unlabeled_imgs = unlabeled_imgs.cuda()
                labels = labels.cuda()
            if iter_count is not 0 and iter_count % change_lr_iter == 0:
                for param in optim_vae.param_groups:
                    param['lr'] = param['lr'] * 0.85
                for param in optim_discriminator.param_groups:
                    param['lr'] = param['lr'] * 0.85
            # VAE step
            for count in range(self.args.num_vae_steps):
                recon, z, mu, logvar, pred_label = vae(labeled_imgs, labeled = 1)
                labeled_task_loss = self.ce_loss(pred_label, labels)
                unsup_loss = self.vae_loss(labeled_imgs, recon, mu, logvar, self.args.beta)
                unlab_recon, unlab_z, unlab_mu, unlab_logvar = vae(unlabeled_imgs, labeled = 0)
                transductive_loss = self.vae_loss(unlabeled_imgs, 
                        unlab_recon, unlab_mu, unlab_logvar, self.args.beta)
            
                labeled_preds = discriminator(mu)
                unlabeled_preds = discriminator(unlab_mu)
                
                lab_real_preds = torch.ones(labeled_imgs.size(0)).cuda()
                unlab_real_preds = torch.ones(unlabeled_imgs.size(0)).cuda()

                dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + self.bce_loss(unlabeled_preds, unlab_real_preds)
                total_vae_loss = unsup_loss + transductive_loss + self.args.adversary_param * dsc_loss +5* labeled_task_loss
                optim_vae.zero_grad()
                total_vae_loss.backward()
                optim_vae.step()

                # sample new batch if needed to train the adversarial network
                if count < (self.args.num_vae_steps - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs, indices = next(unlabeled_data)
                    indices = indices.cpu().numpy()

                    if self.args.cuda:
                        labeled_imgs = labeled_imgs.cuda()
                        unlabeled_imgs = unlabeled_imgs.cuda()
                        labels = labels.cuda()

            # Discriminator step
            for count in range(self.args.num_adv_steps):
                with torch.no_grad():
                    _, _, mu, _ = vae(labeled_imgs, labeled = 0)
                    _, _, unlab_mu, _ = vae(unlabeled_imgs, labeled = 0)
                
                labeled_preds = discriminator(mu)
                unlabeled_preds = discriminator(unlab_mu)
                
                #Relabeling the state of unlabeled samples, unlab_fake_preds is relabeled
                score = []
                for i in range(len(indices)):
                    score.append(scoredic[indices[i]])
                # the score is the new state for unlabeled samples
                lab_real_preds = torch.zeros(labeled_imgs.size(0)).cuda()
                # replace the original state binary 1 with the relabeled state
                unlab_fake_preds = torch.tensor(score, dtype=torch.float).cuda() 

                dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + self.bce_loss(unlabeled_preds, unlab_fake_preds)

                optim_discriminator.zero_grad()
                dsc_loss.backward()
                optim_discriminator.step()

                # sample new batch if needed to train the adversarial network
                if count < (self.args.num_adv_steps - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs, indices = next(unlabeled_data)
                    indices = indices.cpu().numpy()

                    if self.args.cuda:
                        labeled_imgs = labeled_imgs.cuda()
                        unlabeled_imgs = unlabeled_imgs.cuda()
                        labels = labels.cuda()

            if iter_count % 5000 == 0:
                print('  Current training iteration: {}'.format(iter_count) )
                print('  Current vae model loss: {:.4f} {:.4f} '.format(total_vae_loss.item() , labeled_task_loss.item() ))
                print('  Current discriminator model loss: {:.4f}'.format(dsc_loss.item()) )
        return vae, discriminator


    def sample_for_labeling(self, vae, discriminator, unlabeled_dataloader):
        querry_indices = self.sampler.sample(vae, 
                                             discriminator, 
                                             unlabeled_dataloader, 
                                             self.args.cuda)

        return querry_indices
                

    def test(self, task_model):
        task_model.eval()
        total, correct = 0, 0
        for imgs, labels in self.test_dataloader:
            if self.args.cuda:
                imgs = imgs.cuda()

            with torch.no_grad():
                preds = task_model(imgs)

            preds = torch.argmax(preds, dim=1).cpu().numpy()
            correct += accuracy_score(labels, preds, normalize=False)
            total += imgs.size(0)
        return correct / total * 100


    def vae_loss(self, x, recon, mu, logvar, beta):
        MSE = 5*self.mse_loss(recon, x)
        KLD = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar, 1))
        KLD = KLD * beta
        return (MSE + KLD)
