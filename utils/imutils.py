# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:29:03 2020

@author: sshss

usual img-tools writen by huangxin

"""

from numpy import fft
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 图像显示
def displayimg(img, winname):
    cv2.namedWindow(winname,cv2.WINDOW_NORMAL)
    cv2.imshow(winname, img)

# sobel算子
def calcsobel(gray, ksize, weightX, weightY):
    x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize = ksize)
    y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize = ksize)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    sobel = cv2.addWeighted(absX,weightX,absY,weightY,0)
    return sobel

# 生成灰度直方图
def plot_hist(img):
    chans = cv2.split(img)
    colors = ("b","g","r")
    plt.figure()
    plt.xlim([0,256])
    plt.xlabel("Bins")
    plt.ylabel("# of pixels")
    for (color,chan) in zip(colors,chans):
        hist = cv2.calcHist([chan], [0], None, [256], [0,256])
        plt.plot(hist,color=color)

# 图像normalize
def normalize(image, Omin, Omax):
    Imax = np.max(image)
    Imin = np.min(image)
    a = float(Omax-Omin)/(Imax-Imin)
    b = Omin - a*Imin
    dst = a*image + b
    dst = dst.astype("uint8")
    return dst

# 图像的伽马变换，对比度增强
def gamma(image,coef):
    fimg = image/255.0
    dst = np.power(fimg,coef)*255
    dst = np.round(dst)
    dst = dst.astype('uint8')
    return dst

# 阈值自适应canny
def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0,(1.0-sigma)*v))
    upper = int(min(255,(1.0+sigma)*v))
    edged = cv2.Canny(image,lower,upper)
    return edged

# 找轮廓的质心
def find_centroid(contour):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX,cY

# 旋转图片，如果是横图则旋转180度，如果是竖图则顺时针旋转90度
def rotate(img):
    img_x, img_y = img.shape[:2]
    if (img_x < img_y):
        dst = img[::-1,::-1]
    else:
        trans_img = cv2.transpose(img)
        dst = cv2.flip(trans_img, 1)
    return dst

# 维纳滤波器
def wiener(input,PSF,eps,K=0.025):
    input_fft=fft.fft2(input)
    PSF_fft=fft.fft2(PSF) +eps
    PSF_fft_1=np.conj(PSF_fft) /(np.abs(PSF_fft)**2 + K)
    result=fft.ifft2(input_fft * PSF_fft_1)
    result=np.abs(fft.fftshift(result))
    result = result.astype("uint8")
    return result

# 点扩散函数估计
def calcPSF(img_h,img_w,R,halo):
    h = np.zeros((img_h,img_w))
    h = cv2.circle(h, (int(img_w/2),int(img_h/2)), R, 255, halo)
    summa = np.sum(h)
    return h/summa

# 光补偿算法
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

# OTSU阈值分割算法
def otsu_seg(gray, ksize=3):
    # flatten = unevenLightCompensate(gray, 25, (3,3), 0.5)
    sobel = calcsobel(gray, 3, 0.5, 0.5)
    th,dst = cv2.threshold(sobel,0,255,cv2.THRESH_OTSU)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize,ksize))
    dst = cv2.medianBlur(dst, ksize)
    # dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel, iterations=1)
    return dst




