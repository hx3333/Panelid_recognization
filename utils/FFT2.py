# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:49:05 2020

@author: sshss

Fast Fourier Transform 2D

"""

from numpy import fft
import numpy as np
import cv2
import math


def fft2Image(src):
    r,c = src.shape[:2]
    rpadded = cv2.getOptimalDFTSize(r)
    cpadded = cv2.getOptimalDFTSize(c)
    fft2 = np.zeros((rpadded,cpadded,2),np.float32)
    fft2[:r,:c,0] = src
    cv2.dft(fft2,fft2,cv2.DFT_COMPLEX_OUTPUT)
    return fft2

#傅里叶谱图
def amplitudeSpectrum(fft2):
    real2 = np.power(fft2[:,:,0],2.0)
    Imag2 = np.power(fft2[:,:,1],2.0)
    amplitude = np.sqrt(real2+Imag2)
    return amplitude

#傅里叶谱图灰度化显示
def grayamplitude(amplitude):
    amplitude = np.log(amplitude+1.0)
    spectrum = np.zeros(amplitude.shape,np.float32)
    cv2.normalize(amplitude, spectrum,0,1,cv2.NORM_MINMAX)
    return spectrum

#点估计函数
def calcPSF(img_h,img_w,R):
    h = np.zeros((img_h,img_w))
    h = cv2.circle(h, (int(img_h/2),int(img_w/2)), R, 255,8)
    summa = np.sum(h)
    return h/summa

#相位谱
def phaseSpectrum(fft2):
    rows,cols = fft2.shape[:2]
    phase = np.arctan2(fft2[:,:,1],fft2[:,:,0])
    spectrum = phase/math.pi*180
    return spectrum

#中心化
def fftshift(input):
    rows,cols = input.shape[:2]
    image = np.copy(input)
    image = image.astype(np.float32)
    for r in range(rows):
        for c in range(cols):
            if (r+c)%2:
                image[r][c] = (-1)*input[r][c]
    fft2 = fft2Image(image)
    return fft2

def display(winname ,img):
    cv2.namedWindow(winname,cv2.WINDOW_NORMAL)
    cv2.imshow(winname,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def createLPFilter(shape,center,radius,lpType=2,n=2):
    rows,cols=shape[:2]
    r,c = np.mgrid[0:rows:1,0:cols:1]
    c -= center[0]
    r -= center[1]
    d = np.power(c,2.0)+np.power(r,2.0)
    
    lpFilter = np.zeros(shape,np.float32)
    if(radius<=0):
        return lpFilter
    if(lpType ==0):
        lpFilter = np.copy(d)
        lpFilter[lpFilter<pow(radius,2.0)]=1
        lpFilter[lpFilter>=pow(radius,2.0)]=0
    elif(lpType==1):
        lpFilter = 1.0/(1.0+np.power(np.sqrt(d)/radius,2*n))
    elif(lpType==2):
        lpFilter = np.exp(-d/(2.0*pow(radius,2.0)))
        
    return lpFilter
    
    
# lptype=0
# MAX_LPTYPE=2
# radius = 50
# MAX_RADIUS=100

# filename = r'C:\yolo\pytorch-yolo-v3\imgs\dog.jpg'
# img = cv2.imread(filename,0)
# fft2 = fftshift(img)
# # fft2 = fftshift(fft2)

# amplitude = amplitudeSpectrum(fft2)
# spectrum = grayamplitude(amplitude)
# minVal, MaxVal, minLoc, maxLoc = cv2.minMaxLoc(spectrum)
# cv2.namedWindow('lpFilterSpectrum',1)
# def nothing(*arg):
#     pass

# cv2.createTrackbar("lptype", "lpFilterSpectrum", lptype, MAX_LPTYPE, nothing)
# cv2.createTrackbar("radius", "lpFilterSpectrum", radius, MAX_RADIUS, nothing)

# result = np.zeros(spectrum.shape,np.float32)
# while True:
#     radius = cv2.getTrackbarPos("radius", "lpFilterSpectrum")
#     lptype = cv2.getTrackbarPos("lptype", "lpFilterSpectrum")
    
#     lpFilter = createLPFilter(spectrum.shape,maxLoc,radius,lptype)
    
#     rows,cols = spectrum.shape[:2]
#     fImagefft2_lpFilter = np.zeros(fft2.shape, fft2.dtype)
#     for i in range(2):
#         fImagefft2_lpFilter[:rows,:cols,i] = fft2[:rows,:cols,i]*lpFilter
#     lp_amplitude = amplitudeSpectrum(fImagefft2_lpFilter)
#     lp_spectrum = grayamplitude(lp_amplitude)
#     cv2.imshow('lp_spectrum', lp_spectrum)
    
#     cv2.dft(fImagefft2_lpFilter,result,cv2.DFT_REAL_OUTPUT+cv2.DFT_INVERSE+cv2.DFT_SCALE)
#     for r in range(rows):
#         for c in range(cols):
#             if(r+c)%2:
#                 result[r][c]*=-1
#     for r in range(rows):
#         for c in range(cols):
#             if result[r][c]<0:
#                 result[r][c]=0
#             if result[r][c]>255:
#                 result[r][c]=255
#     lpResult = result.astype('uint8')
#     lpResult = lpResult[:img.shape[0],:img.shape[1]]
#     cv2.imshow("LPFilter", lpResult)
#     if cv2.waitKey(5) & 0xFF == ord('q'):
#         break
    
# cv2.destroyAllWindows()

if __name__ == '__main__':
    print('fft2')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



















