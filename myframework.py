#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 15:22:25 2020

@author: lu
"""

import torch
import torch.nn as nn
from torch.autograd import Variable as V

class MyFrame():
    def __init__(self, net, loss, lr=3e-4, evalmode = False):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr, weight_decay=1e-3)
        self.loss = loss()
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()
        
    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id

    def test_batch(self):
        self.forward(volatile=True)
        mask, prec = self.net.forward(self.img)
        mask = mask.cpu().data.numpy().squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        return mask, self.img_id

    def forward(self, volatile=False):
        self.img = V(self.img.cuda(), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.cuda(), volatile=volatile)
        
    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred , prec= self.net.forward(self.img)
        loss = self.loss(self.mask, pred)
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def save(self, path):
        torch.save(self.net.state_dict(), path, _use_new_zipfile_serialization=False)
        
    def load(self, path):
        self.net.load_state_dict(torch.load(path))
    
    def update_lr(self, new_lr, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr
