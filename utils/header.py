# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 17:02:41 2020

@author: sshss
"""
from numpy import fft
import numpy as np
import cv2
from pylibdmtx.pylibdmtx import decode
from PIL import Image
import configparser
import matplotlib.pyplot as plt



def wiener(input,PSF,eps,K=0.025):
    input_fft=fft.fft2(input)
    PSF_fft=fft.fft2(PSF) +eps
    PSF_fft_1=np.conj(PSF_fft) /(np.abs(PSF_fft)**2 + K)
    result=fft.ifft2(input_fft * PSF_fft_1)
    result=np.abs(fft.fftshift(result))
    result = result.astype("uint8")
    return result

def calcPSF(img_h,img_w,R,halo):
    h = np.zeros((img_h,img_w))
    h = cv2.circle(h, (int(img_h/2),int(img_w/2)), R, 255, halo)
    summa = np.sum(h)
    return h/summa


def RotateClockWise90(img):
    trans_img = cv2.transpose(img)
    new_img = cv2.flip(trans_img, 1)
    return new_img
    
def unevenLightCompensate(gray, blockSize, lightcomp_kernel, lightcomp_sigma):
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
    dst = dst.astype("uint8")
    dst = cv2.GaussianBlur(dst, lightcomp_kernel, lightcomp_sigma)
    return dst

def decode_datamatrix(gray,resizeshape,k1,k2):
    th,res = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
    binary = cv2.resize(res, resizeshape, interpolation=cv2.INTER_NEAREST)
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


def filter_noise(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0,255])
    maxLoc = np.where(hist==np.max(hist))
    gray[gray==0] = maxLoc[0][0]
    gray[gray==255] = maxLoc[0][0]
    gray = cv2.medianBlur(gray, 3)  
    return gray

def calcsobel(gray, ksize, weightX, weightY):
    x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize = ksize)    #sobel edge detect
    y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize = ksize)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    sobel = cv2.addWeighted(absX,weightX,absY,weightY,0)
    return sobel

    



