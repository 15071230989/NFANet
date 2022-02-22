#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 18:46:41 2021

@author: lu
"""
import torch
import torch.utils.data as data
from torch.autograd import Variable as V
from torch.utils.data import DataLoader
import cv2
import numpy as np
import os
import math

def randomHorizontalFlip(image, mask, u=0.5):
    '''
    随机水平翻转
    '''
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def randomVerticleFlip(image, mask, u=0.5):
    '''
    随机垂直翻转
    '''
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask

def randomRotate90(image, mask, u=0.5):
    '''
    随机旋转90度
    '''
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)

    return image, mask

def default_loader(id, dir_im, dir_mask):

    img = cv2.imread(os.path.join(dir_im,'{}.jpg').format(id),cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(os.path.join(dir_mask+'{}_gt.png').format(id), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_LINEAR)
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)

    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2,0,1)/255.0
    mask = np.array(mask, np.float32).transpose(2,0,1)/255.0
    mask[mask>=0.5] = 1
    mask[mask<0.5] = 0

    return img, mask


class ImageFolder(data.Dataset):

    def __init__(self, trainlist, dir_im, dir_mask):
        self.ids = trainlist
        self.loader = default_loader
        self.dir_im = dir_im
        self.dir_mask = dir_mask

    def __getitem__(self, index):
        id = self.ids[index]
        img, mask = self.loader(id, self.dir_im, self.dir_mask)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask

    def __len__(self):
        return len(self.ids)



        
        
        
        
        
