# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 10:10:43 2020

@author: sshss

用于分析图片质量的包
"""


import cv2, os
import numpy as np


class getanalysis():
    def __init__(self, img_path, cut_edge):
        self.img_path = img_path
        self.cut_edge = cut_edge
        
    def br_estimate(self, avg_point, display=False):
        '''图像亮度指标：离锚点值，平均偏差'''
        gray = cv2.imread(self.img_path, 0)
        gray = gray[self.cut_edge:-self.cut_edge,self.cut_edge:-self.cut_edge]
        h,w = gray.shape[:2]
        hist = cv2.calcHist([gray], [0], None, [256], [0,256])
        da = np.sum(gray-np.ones((h,w))*avg_point)/(h*w)
        D = np.abs(da)
        mean_error = []
        for i in range(256):
            mean_error.append(abs((i-avg_point)-da)*hist[i])
        ma = np.sum(mean_error)/(h*w)
        M = np.abs(ma)
        k = D/M
        # show the image
        if display:
            cv2.imshow("temp",gray)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return da, k
    
    def blur_estimate(self, threshold, display=False):
        '''利用拉普拉斯算子计算图像模糊指标fm'''
        img = cv2.imread(self.img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray[self.cut_edge:-self.cut_edge,self.cut_edge:-self.cut_edge]
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        if fm < threshold:
            result = "Blurry"
        else:
            result = "UnBlurry"
        # show the image
        if display:       
            cv2.putText(img, "{}: {:.2f}".format(result, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
            cv2.imshow("Image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
        return fm, result
    
    def image_colorfulness(self):
        '''计算图片全局色彩度，Hasler'''
        image = cv2.imread(self.img_path)
        (B, G, R) = cv2.split(image.astype("float")) 
        #rg = R - G
        rg = np.absolute(R - G) 
        #yb = 0.5 * (R + G) - B
        yb = np.absolute(0.5 * (R + G) - B) 
        #calc rg and yb std
        (rbMean, rbStd) = (np.mean(rg), np.std(rg)) 
        (ybMean, ybStd) = (np.mean(yb), np.std(yb)) 
        #calc avg and std
        stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
        meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2)) 
        return stdRoot + (0.3*meanRoot)
    
    def unevenLightCompensate(self, blockSize):
        '''计算图片全局亮度均一性'''
        gray = cv2.imread(self.img_path, 0)
        gray = gray[self.cut_edge:-self.cut_edge,self.cut_edge:-self.cut_edge]
        average = np.mean(gray)   
        rows_new = int(np.ceil(gray.shape[0] / blockSize))
        cols_new = int(np.ceil(gray.shape[1] / blockSize))
        blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
        for r in range(rows_new):
            for c in range(cols_new):
                rowmin = r * blockSize
                rowmax = (r + 1) * blockSize
                if (rowmax > gray.shape[0]):
                    rowmax = gray.shape[0]
                colmin = c * blockSize
                colmax = (c + 1) * blockSize
                if (colmax > gray.shape[1]):
                    colmax = gray.shape[1]
                imageROI = gray[rowmin:rowmax, colmin:colmax]
                temaver = np.mean(imageROI)
                blockImage[r, c] = temaver
        blockImage = blockImage - average
        blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
        gray2 = gray.astype(np.float32)
        dst = gray2 - blockImage2
        
        return dst