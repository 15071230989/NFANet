# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:31:10 2019

@author: ASUS
"""
import cv2
import os
import numpy as np

class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
    def _fast_hist(self, label_pred, label_true):

        mask = (label_true >= 0) & (label_true < self.num_classes)        
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def evaluate(self, predictions, gts):
       
        for lp, lt in zip(predictions, gts):
           
            assert len(lp.flatten()) == len(lt.flatten())
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())    
        # miou
        iou = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        miou = np.nanmean(iou) 
        # dice
        dice = 2 * np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0))
        mdice = np.nanmean(dice)
        # -----------------其他指标------------------------------
        # mean acc
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.nanmean(np.diag(self.hist) / self.hist.sum(axis=1))
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
        return acc, acc_cls, iou, miou, dice, mdice, fwavacc




