
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.utils as vutils

import data
import config
import mnist_model
import pixelcnn
import pixelcnn_loss

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

        self.dis = mnist_model.Discriminative(noise_size=config.noise_size, num_label=config.num_label).cuda()
        self.gen = mnist_model.Generator(image_size=config.image_size, noise_size=config.noise_size).cuda()

        self.dis_optimizer = optim.Adam(self.dis.parameters(), lr=config.dis_lr, betas=(0.5, 0.9999))
        self.gen_optimizer = optim.Adam(self.gen.parameters(), lr=config.gen_lr, betas=(0.0, 0.9999))

        self.pixelcnn = pixelcnn.PixelCNN(nr_resnet=3, disable_third=True, dropout_p=0.0, n_channel=1, image_wh=28).cuda()
        self.pixelcnn.load_state_dict(torch.load(config.pixelcnn_path))

        self.d_criterion = nn.CrossEntropyLoss()

        self.labeled_loader, self.unlabeled_loader, self.unlabeled_loader2, self.dev_loader, self.special_set = data.get_mnist_loaders(config)

        if not os.path.exists(self.config.save_dir):
            os.makedirs(self.config.save_dir)

        log_probs_list = []
        for dev_images, _ in self.dev_loader.get_iter():
            dev_images = Variable(dev_images.cuda(), volatile=True)
            dev_images = (dev_images - 0.5) / 0.5
            dev_images = dev_images.view(-1, 1, 28, 28)
            logits = self.pixelcnn(dev_images)
            log_probs = - pixelcnn_loss.discretized_mix_logistic_loss_c1(dev_images.permute(0, 2, 3, 1), logits.permute(0, 2, 3, 1), sum_all=False)
            log_probs = log_probs.data.cpu().numpy()
            log_probs_list.append(log_probs)
        log_probs = np.concatenate(log_probs_list, axis=0)

        self.unl_ploss_stats = log_probs.min(), log_probs.max(), log_probs.mean(), log_probs.var()
        cut_point = int(log_probs.shape[0] * 0.1)
        self.ploss_th = float(np.partition(log_probs, cut_point)[cut_point])
        print 'ploss_th', self.ploss_th

        print self.dis

    def _get_vis_images(self, labels):
        labels = labels.data.cpu()
        vis_images = self.special_set.index_select(0, labels)
        return vis_images

    def _train(self, labeled=None, vis=False, iter=0):
        config = self.config
        self.dis.train()
        self.gen.train()

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

        unl_logsumexp = log_sum_exp(unl_logits)
        gen_logsumexp = log_sum_exp(gen_logits)

        unl_acc = torch.mean(nn.functional.sigmoid(unl_logsumexp.detach()).gt(0.5).float())
        gen_acc = torch.mean(nn.functional.sigmoid(gen_logsumexp.detach()).gt(0.5).float())

        # This is the typical GAN cost, where sumexp(logits) is seen as the input to the sigmoid
        true_loss = - 0.5 * torch.mean(unl_logsumexp) + 0.5 * torch.mean(F.softplus(unl_logsumexp))
        fake_loss = 0.5 * torch.mean(F.softplus(gen_logsumexp))
        unl_loss = true_loss + fake_loss
         
        d_loss = lab_loss + unl_loss

        self.dis_optimizer.zero_grad()
        d_loss.backward()
        self.dis_optimizer.step()

        ##### train Gen and Enc
        unl_images, _ = self.unlabeled_loader2.next()
        unl_images = Variable(unl_images.cuda())
        noise = Variable(torch.Tensor(unl_images.size(0), config.noise_size).uniform_().cuda())
        gen_images = self.gen(noise)

        unl_feat = self.dis(unl_images, feat=True)
        gen_feat = self.dis(gen_images, feat=True)
        fm_loss = torch.mean((torch.mean(gen_feat, 0) - torch.mean(unl_feat, 0)) ** 2)

        gen_loss = fm_loss

        if iter > 9000 and random.random() < config.p_loss_prob:
            noise = Variable(torch.Tensor(30, config.noise_size).uniform_().cuda())
            gen_images = self.gen(noise)
            gen_images = (gen_images - 0.5) / 0.5
            gen_images = gen_images.view(-1, 1, 28, 28)
            logits = self.pixelcnn(gen_images)
            log_probs = - pixelcnn_loss.discretized_mix_logistic_loss_c1(gen_images.permute(0, 2, 3, 1), logits.permute(0, 2, 3, 1), sum_all=False)
            p_loss = torch.max(log_probs - self.ploss_th, Variable(torch.cuda.FloatTensor(log_probs.size()).fill_(0.0)))
            non_zero_cnt = float((p_loss > 0).sum().data.cpu()[0])
            if non_zero_cnt > 0:
                p_loss = p_loss.sum() / non_zero_cnt * config.p_loss_weight
            else:
                p_loss = 0
        else:
            p_loss = 0

        loss = gen_loss + p_loss

        self.gen_optimizer.zero_grad()
        loss.backward()
        self.gen_optimizer.step()

        monitor_dict = OrderedDict([
                       ('unl accuracy' , unl_acc.data[0]), 
                       ('gen accuracy' , gen_acc.data[0]), 
                       ('lab loss' , lab_loss.data[0]), 
                       ('unl loss' , unl_loss.data[0]),
                       ('true loss' , true_loss.data[0]), 
                       ('fake loss' , fake_loss.data[0]), 
                       ('gen loss' , gen_loss.data[0]), 
                       ('p loss', p_loss.data[0] if hasattr(p_loss, 'data') else 0.0)
                   ])
                
        return monitor_dict

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
        gen_images = gen_images.view(-1, 1, 28, 28)

        save_path = os.path.join(self.config.save_dir, 'FM+LD.{}.png'.format(self.config.suffix))
        vutils.save_image(gen_images.data.cpu(), save_path, normalize=False, range=(-1,1), nrow=10)

    def param_init(self):
        def func_gen(flag):
            def func(m):
                if hasattr(m, 'init_mode'):
                    setattr(m, 'init_mode', flag)
            return func

        images = []
        for i in range(500 / self.config.train_batch_size):
            unl_images, _ = self.unlabeled_loader.next()
            images.append(unl_images)
        images = torch.cat(images, 0)

        self.dis.apply(func_gen(True))
        logits = self.dis(Variable(images.cuda()))
        self.dis.apply(func_gen(False))

    def train(self):
        config = self.config

        print config.train_batch_size % len(self.unlabeled_loader)
        self.param_init()

        self.iter_cnt = 0
        iter, min_dev_incorrect = 0, 1e6
        monitor = OrderedDict()

        batch_per_epoch = int((len(self.unlabeled_loader) + config.train_batch_size - 1) / config.train_batch_size)
        while True:

            if iter % batch_per_epoch == 0:
                epoch = iter / batch_per_epoch
                if epoch >= config.max_epochs:
                    break
                epoch_ratio = float(epoch) / float(config.max_epochs)
                # use another outer max to prevent any float computation precision problem
                self.dis_optimizer.param_groups[0]['lr'] = config.dis_lr * max(0., min(3. * (1. - epoch_ratio), 1.))
                self.gen_optimizer.param_groups[0]['lr'] = config.gen_lr * max(0., min(3. * (1. - epoch_ratio), 1.))

            iter_vals = self._train(iter=iter)

            for k, v in iter_vals.items():
                if not monitor.has_key(k):
                    monitor[k] = 0.
                monitor[k] += v

            if iter % config.vis_period == 0:
                self.visualize()

            if iter % config.eval_period == 0:
                train_loss, train_incorrect = self.eval(self.labeled_loader)
                dev_loss, dev_incorrect = self.eval(self.dev_loader)

                min_dev_incorrect = min(min_dev_incorrect, dev_incorrect)
                disp_str = '#{}\ttrain: {:.4f}, {} | dev: {:.4f}, {} | best: {}'.format(
                        iter, train_loss, train_incorrect, dev_loss, dev_incorrect, min_dev_incorrect)
                for k, v in monitor.items():
                    disp_str += ' | {}: {:.4f}'.format(k, v / config.eval_period)

                disp_str += ' | lr: dis {:.5f}, gen {:.5f}'.format(
                    self.dis_optimizer.param_groups[0]['lr'], self.gen_optimizer.param_groups[0]['lr'])
                monitor = OrderedDict()

                print disp_str

                noise = Variable(torch.Tensor(400, self.config.noise_size).uniform_().cuda(), volatile=True)
                images = self.gen(noise)
                images = (images - 0.5) / 0.5
                images = images.view(-1, 1, 28, 28)
                logits = self.pixelcnn(images)
                log_probs = - pixelcnn_loss.discretized_mix_logistic_loss_c1(images.permute(0, 2, 3, 1), logits.permute(0, 2, 3, 1), sum_all=False).data.cpu()
                gen_ploss_stats = log_probs.min(), log_probs.max(), log_probs.mean(), log_probs.var()
                print 'gen stats', gen_ploss_stats
                print 'unl stats', self.unl_ploss_stats

            iter += 1
            self.iter_cnt += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='mnist_trainer.py')
    parser.add_argument('-suffix', default='run0', type=str, help="Suffix added to the save images.")

    args = parser.parse_args()

    trainer = Trainer(config.mnist_config(), args)
    trainer.train()


