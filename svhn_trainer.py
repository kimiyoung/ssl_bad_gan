
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.utils as vutils

import data
import config
import model

import random
import time
import os, sys
import math
import argparse
from collections import OrderedDict

import numpy as np
from utils import *

class Trainer(object):

    def __init__(self, config, args):
        self.config = config
        for k, v in args.__dict__.items():
            setattr(self.config, k, v)
        setattr(self.config, 'save_dir', '{}_log'.format(self.config.dataset))

        disp_str = ''
        for attr in sorted(dir(self.config), key=lambda x: len(x)):
            if not attr.startswith('__'):
                disp_str += '{} : {}\n'.format(attr, getattr(self.config, attr))
        sys.stdout.write(disp_str)
        sys.stdout.flush()

        self.labeled_loader, self.unlabeled_loader, self.unlabeled_loader2, self.dev_loader, self.special_set = data.get_svhn_loaders(config)

        self.dis = model.Discriminative(config).cuda()
        self.gen = model.Generator(image_size=config.image_size, noise_size=config.noise_size).cuda()

        self.dis_optimizer = optim.Adam(self.dis.parameters(), lr=config.dis_lr, betas=(0.5, 0.999)) # 0.0 0.9999
        self.gen_optimizer = optim.Adam(self.gen.parameters(), lr=config.gen_lr, betas=(0.0, 0.999)) # 0.0 0.9999

        self.d_criterion = nn.CrossEntropyLoss()

        if not os.path.exists(self.config.save_dir):
            os.makedirs(self.config.save_dir)

        log_path = os.path.join(self.config.save_dir, '{}.FM+PT+ENT.{}.txt'.format(self.config.dataset, self.config.suffix))
        self.logger = open(log_path, 'wb')
        self.logger.write(disp_str)

    def _get_vis_images(self, labels):
        labels = labels.data.cpu()
        vis_images = self.special_set.index_select(0, labels)
        return vis_images

    def _train(self, labeled=None, vis=False):
        config = self.config
        self.dis.train()
        self.gen.train()

        ##### train Dis
        lab_images, lab_labels = self.labeled_loader.next()
        lab_images, lab_labels = Variable(lab_images.cuda()), Variable(lab_labels.cuda())

        unl_images, _ = self.unlabeled_loader.next()
        unl_images = Variable(unl_images.cuda())

        noise = Variable(torch.Tensor(unl_images.size(0), config.noise_size).uniform_().cuda())
        gen_images = self.gen(noise)
        
        lab_logits = self.dis(lab_images)
        unl_logits = self.dis(unl_images)
        gen_logits = self.dis(gen_images.detach())

        # Standard classification loss
        lab_loss = self.d_criterion(lab_logits, lab_labels)

        # Conditional entropy loss
        ent_loss = config.ent_weight * entropy(unl_logits)

        # GAN true-fake loss: sumexp(logits) is seen as the input to the sigmoid
        unl_logsumexp = log_sum_exp(unl_logits)
        gen_logsumexp = log_sum_exp(gen_logits)

        true_loss = - 0.5 * torch.mean(unl_logsumexp) + 0.5 * torch.mean(F.softplus(unl_logsumexp))
        fake_loss = 0.5 * torch.mean(F.softplus(gen_logsumexp))
        unl_loss = true_loss + fake_loss
         
        d_loss = lab_loss + unl_loss + ent_loss

        ##### Monitoring (train mode)
        # true-fake accuracy
        unl_acc = torch.mean(nn.functional.sigmoid(unl_logsumexp.detach()).gt(0.5).float())
        gen_acc = torch.mean(nn.functional.sigmoid(gen_logsumexp.detach()).gt(0.5).float())
        # top-1 logit compared to 0: to verify Assumption (2) and (3)
        max_unl_acc = torch.mean(unl_logits.max(1)[0].detach().gt(0.0).float())
        max_gen_acc = torch.mean(gen_logits.max(1)[0].detach().gt(0.0).float())

        self.dis_optimizer.zero_grad()
        d_loss.backward()
        self.dis_optimizer.step()

        ##### train Gen and Enc
        noise = Variable(torch.Tensor(unl_images.size(0), config.noise_size).uniform_().cuda())
        gen_images = self.gen(noise)

        # Feature matching loss
        unl_feat = self.dis(unl_images, feat=True)
        gen_feat = self.dis(gen_images, feat=True)
        fm_loss = torch.mean(torch.abs(torch.mean(gen_feat, 0) - torch.mean(unl_feat, 0)))

        # Entropy loss via feature pull-away term
        nsample = gen_feat.size(0)
        gen_feat_norm = gen_feat / gen_feat.norm(p=2, dim=1).expand_as(gen_feat)
        cosine = torch.mm(gen_feat_norm, gen_feat_norm.t())
        mask = Variable((torch.ones(cosine.size()) - torch.diag(torch.ones(nsample))).cuda())
        pt_loss = config.pt_weight * torch.sum((cosine * mask) ** 2) / (nsample * (nsample-1))

        # Generator loss
        g_loss = fm_loss + pt_loss
        
        self.gen_optimizer.zero_grad()
        g_loss.backward()
        self.gen_optimizer.step()

        monitor_dict = OrderedDict([
                       ('unl acc' , unl_acc.data[0]), 
                       ('gen acc' , gen_acc.data[0]), 
                       ('max unl acc' , max_unl_acc.data[0]), 
                       ('max gen acc' , max_gen_acc.data[0]), 
                       ('lab loss' , lab_loss.data[0]),
                       ('unl loss' , unl_loss.data[0]),
                       ('ent loss' , ent_loss.data[0]),
                       ('fm loss' , fm_loss.data[0]),
                       ('pt loss' , pt_loss.data[0])
                   ])
                
        return monitor_dict

    def eval_true_fake(self, data_loader, max_batch=None):
        self.gen.eval()
        self.dis.eval()
        
        cnt = 0
        unl_acc, gen_acc, max_unl_acc, max_gen_acc = 0., 0., 0., 0.
        for i, (images, _) in enumerate(data_loader.get_iter()):
            images = Variable(images.cuda(), volatile=True)
            noise = Variable(torch.Tensor(images.size(0), self.config.noise_size).uniform_().cuda(), volatile=True)

            unl_feat = self.dis(images, feat=True)
            gen_feat = self.dis(self.gen(noise), feat=True)

            unl_logits = self.dis.out_net(unl_feat)
            gen_logits = self.dis.out_net(gen_feat)

            unl_logsumexp = log_sum_exp(unl_logits)
            gen_logsumexp = log_sum_exp(gen_logits)

            ##### Monitoring (eval mode)
            # true-fake accuracy
            unl_acc += torch.mean(nn.functional.sigmoid(unl_logsumexp).gt(0.5).float()).data[0]
            gen_acc += torch.mean(nn.functional.sigmoid(gen_logsumexp).gt(0.5).float()).data[0]
            # top-1 logit compared to 0: to verify Assumption (2) and (3)
            max_unl_acc += torch.mean(unl_logits.max(1)[0].gt(0.0).float()).data[0]
            max_gen_acc += torch.mean(gen_logits.max(1)[0].gt(0.0).float()).data[0]

            cnt += 1
            if max_batch is not None and i >= max_batch - 1: break

        return unl_acc / cnt, gen_acc / cnt, max_unl_acc / cnt, max_gen_acc / cnt

    def eval(self, data_loader, max_batch=None):
        self.gen.eval()
        self.dis.eval()

        loss, incorrect, cnt = 0, 0, 0
        for i, (images, labels) in enumerate(data_loader.get_iter()):
            images = Variable(images.cuda(), volatile=True)
            labels = Variable(labels.cuda(), volatile=True)
            pred_prob = self.dis(images)
            loss += self.d_criterion(pred_prob, labels).data[0]
            cnt += 1
            incorrect += torch.ne(torch.max(pred_prob, 1)[1], labels).data.sum()
            if max_batch is not None and i >= max_batch - 1: break
        return loss / cnt, incorrect


    def visualize(self):
        self.gen.eval()
        self.dis.eval()

        vis_size = 100
        noise = Variable(torch.Tensor(vis_size, self.config.noise_size).uniform_().cuda())
        gen_images = self.gen(noise)

        save_path = os.path.join(self.config.save_dir, '{}.FM+PT+Ent.{}.png'.format(self.config.dataset, self.config.suffix))
        vutils.save_image(gen_images.data.cpu(), save_path, normalize=True, range=(-1,1), nrow=10)

    def param_init(self):
        def func_gen(flag):
            def func(m):
                if hasattr(m, 'init_mode'):
                    setattr(m, 'init_mode', flag)
            return func

        images = []
        for i in range(500 / self.config.train_batch_size):
            lab_images, _ = self.labeled_loader.next()
            images.append(lab_images)
        images = torch.cat(images, 0)

        self.gen.apply(func_gen(True))
        noise = Variable(torch.Tensor(images.size(0), self.config.noise_size).uniform_().cuda())
        gen_images = self.gen(noise)
        self.gen.apply(func_gen(False))

        self.dis.apply(func_gen(True))
        logits = self.dis(Variable(images.cuda()))
        self.dis.apply(func_gen(False))

    def train(self):
        config = self.config
        self.param_init()

        self.iter_cnt = 0
        iter, min_dev_incorrect = 0, 1e6
        monitor = OrderedDict()
        
        batch_per_epoch = int((len(self.unlabeled_loader) + config.train_batch_size - 1) / config.train_batch_size)
        min_lr = config.min_lr if hasattr(config, 'min_lr') else 0.0
        while True:

            if iter % batch_per_epoch == 0:
                epoch = iter / batch_per_epoch
                if config.dataset != 'svhn' and epoch >= config.max_epochs:
                    break
                epoch_ratio = float(epoch) / float(config.max_epochs)
                # use another outer max to prevent any float computation precision problem
                self.dis_optimizer.param_groups[0]['lr'] = max(min_lr, config.dis_lr * min(3. * (1. - epoch_ratio), 1.))
                self.gen_optimizer.param_groups[0]['lr'] = max(min_lr, config.gen_lr * min(3. * (1. - epoch_ratio), 1.))

            iter_vals = self._train()

            for k, v in iter_vals.items():
                if not monitor.has_key(k):
                    monitor[k] = 0.
                monitor[k] += v

            if iter % config.vis_period == 0:
                self.visualize()

            if iter % config.eval_period == 0:
                train_loss, train_incorrect = self.eval(self.labeled_loader)
                dev_loss, dev_incorrect = self.eval(self.dev_loader)

                unl_acc, gen_acc, max_unl_acc, max_gen_acc = self.eval_true_fake(self.dev_loader, 10)

                train_incorrect /= 1.0 * len(self.labeled_loader)
                dev_incorrect /= 1.0 * len(self.dev_loader)
                min_dev_incorrect = min(min_dev_incorrect, dev_incorrect)

                disp_str = '#{}\ttrain: {:.4f}, {:.4f} | dev: {:.4f}, {:.4f} | best: {:.4f}'.format(
                    iter, train_loss, train_incorrect, dev_loss, dev_incorrect, min_dev_incorrect)
                for k, v in monitor.items():
                    disp_str += ' | {}: {:.4f}'.format(k, v / config.eval_period)
                
                disp_str += ' | [Eval] unl acc: {:.4f}, gen acc: {:.4f}, max unl acc: {:.4f}, max gen acc: {:.4f}'.format(unl_acc, gen_acc, max_unl_acc, max_gen_acc)
                disp_str += ' | lr: {:.5f}'.format(self.dis_optimizer.param_groups[0]['lr'])
                disp_str += '\n'

                monitor = OrderedDict()

                self.logger.write(disp_str)
                sys.stdout.write(disp_str)
                sys.stdout.flush()

            iter += 1
            self.iter_cnt += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='svhn_trainer.py')
    parser.add_argument('-suffix', default='run0', type=str, help="Suffix added to the save images.")

    args = parser.parse_args()

    trainer = Trainer(config.svhn_config(), args)
    trainer.train()
