#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 09:29:05 2021

@author: lu
"""
import cv2
import os
import numpy as np

dir_im = './res/subout/'
dir_point = './dataset/POINT/'
target = './res/out/'
deletearea = 40
k = 2
imlist = os.listdir(os.path.join(dir_im, '0_0'))
if not os.path.exists(target):
    os.mkdir(target)

for im in imlist:
    point_label = cv2.imread(os.path.join(dir_point, im))
    point_label = point_label[:, :, 0]
    point_label = cv2.resize(point_label, (492, 492), interpolation=cv2.INTER_LINEAR)
    point = np.zeros(point_label.shape, dtype=np.uint8)

    mean = k * k
    votemap = np.zeros((point_label.shape[0], point_label.shape[1]), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            map = cv2.imread(os.path.join(dir_im + '{}_{}'.format(i, j), im))
            map = map[:, :, 0]
            #ret, map = cv2.threshold(map, 0, 255, cv2.THRESH_OTSU)
            votemap += (map / mean)
    votemap = np.uint8(votemap)
    votemap[votemap >= 127] = 255
    votemap[votemap < 127] = 0

    point_list = []
    point_ctrs, _ = cv2.findContours(point_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for r in range(len(point_ctrs)):
        dist = np.empty(point_label.shape, dtype=np.float32)
        for l in range(point_label.shape[0]):
            for w in range(point_label.shape[1]):
                dist[l, w] = cv2.pointPolygonTest(point_ctrs[r], (w, l), True)
        minval, maxval, _, maxdistpt = cv2.minMaxLoc(dist)
        cx, cy = maxdistpt[0], maxdistpt[1]
        point_list.append((cx, cy))

    # delete area
    new_mask = np.zeros(votemap.shape, np.uint8)
    contours, hierarchy = cv2.findContours(votemap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for j in range(len(contours)):
        area = cv2.contourArea(contours[j])
        if area > deletearea:
            for (x, y) in point_list:
                if cv2.pointPolygonTest(contours[j], (x, y), False) != -1:
                    cv2.drawContours(new_mask, contours[j], -1, (255, 255, 255), thickness=-1)

    # full hole
    contours, hierarchy = cv2.findContours(new_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_list = []
    for r in range(len(contours)):
        drawing = np.zeros_like(new_mask)
        img_contour = cv2.drawContours(drawing, contours, r, (255, 255, 255), thickness=-1)
        contours_list.append(img_contour)
    out = sum(contours_list)
    if len(contours_list) == 0:
        out = np.zeros(votemap.shape, np.uint8)

    out += point_label
    out[out>=127] = 255
    out[out < 127] = 0
    cv2.imwrite(target + '/' + im[:-4] + '.png', out)
    print('------', im, '------')

print('Finish!')