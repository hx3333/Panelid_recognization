# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 11:07:01 2020

@author: sshss
"""


import numpy as np
import cv2
from .imutils import rotate


def kmeans(gray, n_cluster, display=False):
    '''
参数：
	data: 分类数据，最好是np.float32的数据，每个特征放一列。
	K: 分类数，opencv2的kmeans分类是需要已知分类数的。
	bestLabels：预设的分类标签或者None
	criteria：迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
	    其中，type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
         —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
	attempts：重复试验kmeans算法次数，将会返回最好的一次结果
	flags：初始中心选择，有两种方法：
   		——v2.KMEANS_PP_CENTERS;
    	——cv2.KMEANS_RANDOM_CENTERS

返回值：
	compactness：紧密度，返回每个点到相应重心的距离的平方和
	labels：结果标记，每个成员被标记为0,1等
	centers：由聚类的中心组成的数组
'''
    h,w = gray.shape[:2]
    # 将图像矩阵展平
    gray_flat = gray.reshape((h*w, 1))
    gray_flat = np.float32(gray_flat)
    # 迭代参数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 20, 0.5)
    flags = cv2.KMEANS_RANDOM_CENTERS
    # 聚类
    compactness, labels, centers = cv2.kmeans(gray_flat, 2, None, criteria, 10, flags)
    labels = labels.reshape((h,w))*255
    labels = labels.astype('uint8')
    if display:
        cv2.imshow("seg",labels)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return labels
    

    
    