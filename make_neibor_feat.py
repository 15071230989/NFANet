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

from networks.unet import UNet, UNetNeighbor


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
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
            mask,prec = self.pred_mask(img)
            return mask,prec

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

def getFeatureMaps(prec):
    prec = np.array(prec.cpu().detach().numpy())
    _range = np.max(prec)-np.min(prec)
    prec = np.uint8((prec - np.min(prec)) / _range * 255)
    prec = prec[0,:,:]
    #######color########
    #prec = cv2.applyColorMap(prec, cv2.COLORMAP_JET)
    #######color########
    prec = cv2.resize(prec, (492, 492), interpolation=cv2.INTER_LINEAR)
    return prec

k=2
source = './dataset/tr_im/'
target = './res/neibor_map/'
NETNAME = 'Persduo_net'
val = os.listdir(source)
solver = TTAFrame(UNet)
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
            _, precs = solver.test_one_img_from_path(source+'/'+name)
            end.record()
            torch.cuda.synchronize()

            prec0 =precs[0]
            #prec1 =precs[1]
            #prec2 =precs[2]
            prec0 = getFeatureMaps(prec0)
            #prec1 = getFeatureMaps(prec1)
            #prec2 = getFeatureMaps(prec2)
            #CONCAT = np.concatenate([prec0,prec1,prec2], axis=1)
            cv2.imwrite(target_im+'/'+name[:-4]+'_gt.png',prec0)

print('Finish!')

    
    
