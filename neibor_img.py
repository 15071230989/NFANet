#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 18:46:41 2021

@author: lu
"""

import cv2
import os

def neighbor_sampler(source_im, k=2):
    val = os.listdir(source_im)
    for _, name in enumerate(val):
        index = name[:-4]
        img = cv2.imread(os.path.join(source_im, '{}.jpg').format(index), cv2.IMREAD_UNCHANGED)

        for i in range(k):
            for j in range(k):
                ner_img = img[i::k, j::k,:]
                target_im = source_im + '{}_{}/'.format(i, j)
                if not os.path.exists(target_im):
                    os.mkdir(target_im)
                cv2.imwrite(os.path.join(target_im,'{}.jpg').format(index), ner_img)

source_im = './dataset/tr_im/'

neighbor_sampler(source_im)
