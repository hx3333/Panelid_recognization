# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:29:42 2020

@author: sshss
"""

import numpy as np
import cv2, math
from pylibdmtx.pylibdmtx import decode
from PIL import Image
from .Kmeans_seg import kmeans
from .imutils import calcsobel

class getmatrix():
    '''
    这是一个从预处理图找到目标轮廓的类
    '''
    def __init__(self, cut_edge=3):
        self.cut_edge = cut_edge
    def findcontours(self, seg, size, display=False):
        # seg = kmeans(gray, n_cluster=2, cut_edge=self.cut_edge, display=True)
        # img = gray[self.cut_edge:-self.cut_edge,self.cut_edge:-self.cut_edge]
        contours, hierarchy = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        canvas = seg.copy()
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        for i in contours:
            x,y,w,h = cv2.boundingRect(i)
            aspect = w/h
            img = cv2.rectangle(canvas, (x,y), (x+w,y+h), (0,0,255), 1)
            # format (w_lower,w_upper,h_lower,h_upper)
            if (size[0]<w<size[1]) and (size[2]<h<size[3]) and aspect>0.98:
                matrix_bbox = (x,y,w,h)
            else:
                continue
        if display:
            cv2.imshow('canvas',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        try:
            return matrix_bbox
        except Exception as e:
            return None

def decode_datamatrix(binary, k1, k2):
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, k1)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, k2)
    final = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel1, iterations=1)
    decode_str = decode(Image.fromarray(final))
    if decode_str == []:
        finalretry = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel2, iterations=1)
        decode_str_retry = decode(Image.fromarray(finalretry))
        if decode_str_retry == []:
            return None
        else:
            return str(decode_str_retry[0][0])[2:-1]
    else:
        return str(decode_str[0][0])[2:-1]
    
def get_min_dist(binary, bbox, mode='roof'):
    h, w = binary.shape[:2]
    bbox_x1 = bbox[0]
    bbox_y1 = bbox[1]
    bbox_x2 = bbox_x1 + bbox[2]
    bbox_y2 = bbox_y1 + bbox[3]
    for dist in range(h):
        if mode == 'roof':
            try:
                roof_dist = int(binary[bbox_y1-dist-1,bbox_x1])-int(binary[bbox_y1-dist,bbox_x1])
            except IndexError:
                return bbox_y1
            if roof_dist != 0:
                return dist+1
        elif mode == 'ground':
            try:
                ground_dist = int(binary[bbox_y2+dist+1,bbox_x2])-int(binary[bbox_y2+dist,bbox_x2])
            except IndexError:
                return h-bbox_y2
            if ground_dist != 0:
                return dist+1
        
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    