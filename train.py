#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 09:46:24 2020

@author: lu
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')
from time import time
from networks.unet import UNet,UNetNeighbor
from compute import IOUMetric
from myframework import MyFrame
from loss import dice_bce_loss
import os
import numpy as np
from data import ImageFolder
import cv2

k = 2
dir_im = './dataset/tr_im/'
dir_mask = './dataset/persduo/'
NETNAME = 'recur'

for i in range(k):
    for j in range(k):

        SHAPE = (512,512)
        NAME = NETNAME + '_{}_{}'.format(i,j)
        solver = MyFrame(UNet, dice_bce_loss , 0.0001)

        dir_img =  os.path.join(dir_im + '{}_{}/'.format(i,j))

        imlist = os.listdir(dir_img)
        imlist =list(map(lambda x: x[:-4], imlist))
        datasets = ImageFolder(imlist, dir_im, dir_mask)
        data_loader = DataLoader(datasets, batch_size=4, shuffle=True, num_workers=0)

        tic = time()
        no_optim = 0
        total_epoch = 100
        train_epoch_best_loss = 100.

        for epoch in range(1, total_epoch + 1):
            data_loader_iter = iter(data_loader)
            train_epoch_loss = 0
            train_epoch_dice = 0
            train_epoch_iou = 0
            for img, mask in data_loader_iter:

                solver.set_input(img, mask)
                train_loss = solver.optimize()
                train_epoch_loss += train_loss
                preds = solver.test_batch()
                preds = np.array(preds[0]).astype(np.uint8)
                labels = mask.cpu().data.numpy().squeeze(1)
                el = IOUMetric(2)

                labels[labels>0]=1
                labels[labels<0]=0

                acc, acc_cls, iou, miou, dice, mdice, fwavacc = el.evaluate(preds, labels)
                train_epoch_dice += dice
                train_epoch_iou += iou
            train_epoch_dice /= len(data_loader_iter)
            train_epoch_loss /= len(data_loader_iter)
            train_epoch_iou /= len(data_loader_iter)

            print('********')
            print('epoch:',epoch,'    time:',int(time()-tic))
            print('train_loss:',train_epoch_loss)
            print('train_dice',train_epoch_dice)
            print('train_iou',train_epoch_iou)
            print('SHAPE:',SHAPE)

            if train_epoch_loss >= train_epoch_best_loss:
                no_optim += 1
            else:
                no_optim += 0
                train_epoch_best_loss = train_epoch_loss
                solver.save('weights/'+NAME+'.th')
            if no_optim > 6:
                print('early stop at %d epoch' % epoch)
                break
            if no_optim > 3:
                if solver.old_lr < 5e-6:
                    break
                solver.load('weights/'+NAME+'.th')
                solver.update_lr(5.0, factor = True)

print('Finish!')
