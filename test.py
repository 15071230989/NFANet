#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 18:46:41 2021

@author: lu
"""

import torch
import cv2
import os
import numpy as np

from networks.unet import UNetNeighbor


BATCHSIZE_PER_CARD = 4
class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
    def random_augment(self, img):
        return img
        
    def pred_mask(self, img):
            img = np.array(img, np.float32).transpose(2,0,1)/255.0           
            img = np.expand_dims(img, axis=0)  
            img = torch.Tensor(img)            
#            mask = self.net.forward(img).squeeze().cpu().data.numpy()
            mask,prec= self.net.forward(img)
            mask = mask.squeeze().cpu().data.numpy()   
            return mask,prec
        
    def test_one_img_from_path(self, path, evalmode = True):        
        if evalmode:
            self.net.eval()
            img = cv2.imread(path)
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
            mask,prec = self.pred_mask(img)
            return mask,prec

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

k=2
source = './dataset/tr_im/'
target = './res/subout/'
NETNAME = 'recur'
val = os.listdir(source)
solver = TTAFrame(UNetNeighbor)
NAME = 'weights/' + NETNAME
for i in range(k):
    for j in range(k):
        solver.load(NAME + '_{}_{}.th'.format(i,j))
        target_im = target + '{}_{}/'.format(i, j)
        if not os.path.exists(target_im):
            os.mkdir(target_im)

        for w,name in enumerate(val):
            if name[1:2] == '_':
                continue
            start = torch.cuda.Event(enable_timing=True) #the times
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            mask, precs = solver.test_one_img_from_path(source+'/'+name)
            end.record()
            torch.cuda.synchronize()

            mask = cv2.resize(mask, (492, 492), interpolation=cv2.INTER_LINEAR)
            #mask[mask >= 0.5] = 255
            #mask[mask < 0.5] = 0
            mask *= 255
            mask = np.concatenate([mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=2)
            mask = mask.astype(np.uint8)
            cv2.imwrite(target_im+'/'+name[:-4]+'_gt.png',mask)

print('Finish!')

    
    
