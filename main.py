# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 11:25:25 2020

@author: sshss
"""

import sys, time, datetime, os ,cv2, shutil, logging, random, glob
import configparser
from utils.img_analysis import getanalysis
from utils.imutils import *
from utils.Kmeans_seg import kmeans
from utils.decoding import getmatrix, decode_datamatrix,get_min_dist


def readconfig():
    config = configparser.ConfigParser()
    config.read("./cfg/Pnlid.ini",encoding = "utf-8")
    return config

def filtnone(obj1, obj2):
    if obj1 is None:
        return obj2
    elif obj2 is None:
        return obj1
    else:
        return obj1
    
def initLogging(filename):
    global logger
    logger = logging.getLogger()
    if logger == None:
        logger = logging.getLogger()
    else:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(filename)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    
def main(img_path):
    #initial flag
    flag = 0
    
    cfg = readconfig()
    # img analyser
    blur_thresh = float(cfg["img_analyser"]["blur_thresh"])
    avg_point = int(cfg["img_analyser"]["avg_point"])
    cut_edge = int(cfg["img_analyser"]["cut_edge"])
    overexpo = float(cfg["img_analyser"]["expo_upper_limit"])
    underexpo = float(cfg["img_analyser"]["expo_lower_limit"])
    
    #deblur params
    NSR = float(cfg["wiener"]["NSR"])
    eps = float(cfg["wiener"]["eps"])
    R = int(cfg["PSF"]["radius"])
    halo = int(cfg["PSF"]["halo"])
    
    #decode params
    k1 = int(cfg["morphology"]["k1"])
    k2 = int(cfg["morphology"]["k2"])
    
    #contours params
    matrix_w_lower = int(cfg["contours"]["matrix_w_lower"])
    matrix_w_upper = int(cfg["contours"]["matrix_w_upper"])
    matrix_h_lower = int(cfg["contours"]["matrix_h_lower"])
    matrix_h_upper = int(cfg["contours"]["matrix_h_upper"])
    
    #matrix params
    extend_w = int(cfg["matrix"]["extend_w"])
    extend_h = int(cfg["matrix"]["extend_h"])
    dist_thresh = int(cfg["matrix"]["dist_thresh"])
    
    date = datetime.datetime.now().strftime('%Y%m%d')
    clock = datetime.datetime.now().strftime("%H%M")
    abl_imgs = "./abnormal/"+date
    if not os.path.exists(abl_imgs):
        os.makedirs(abl_imgs)
    logfolder = "./log/Detectlog/"
    if not os.path.exists(logfolder):
        os.makedirs(logfolder)
    log = logfolder + date + '.log'
    initLogging(log)
    # 加载图片并转到正常角度
    img = cv2.imread(img_path)
    if img is None:
        logging.info("Current input Error! File: %s"%img_path)
        logging.info("Cannot load the image ,make sure your filepath is correct!!!")
        flag = 1
        return flag
    img = rotate(img)
    img = img[cut_edge:-cut_edge,cut_edge:-cut_edge]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    # 导入图像分析工具,为了消除边缘干扰切除边缘3pix
    analy = getanalysis(img_path, cut_edge=cut_edge)
    # 亮度判定
    da,K = analy.br_estimate(avg_point)
    # 模糊判定
    fm,blur = analy.blur_estimate(threshold = blur_thresh)
    # 图像分析日志
    logging.info("Analyser start at {}".format(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")))
    logging.info('da:%.2f'%da)
    logging.info('K:%.2f'%K)
    logging.info('fm:{:.2f}, fm thresh:{:.2f}'.format(fm, blur_thresh))
    
    # 限定对比度阈值的自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
    if K >= 1:
        if da > overexpo:
            logging.info("over exposure, ADI cannot recognize")
            shutil.copy(img_path,os.path.join(abl_imgs,img_path.split('\\')[-1]))
            flag = 1
            return flag
        elif da < underexpo:
            logging.info("under exposure, ADI cannot recognize")
            shutil.copy(img_path,os.path.join(abl_imgs,img_path.split('\\')[-1]))
            flag = 1
            return flag
        else:
            pre_process = clahe.apply(gray)
    
    else:
        pre_process = gray
        
    if fm < blur_thresh:
        logging.info("Blurry image, ADI start deblur-algo program")
        PSF = calcPSF(h, w, R, halo)
        pre_process = wiener(gray, PSF, eps, NSR)
    
    pre_process = cv2.medianBlur(pre_process, k2)
    # 用otsu阈值分割和kmeans同时对二维码区域进行分割
    seg = kmeans(pre_process, n_cluster=2)
    otsu = otsu_seg(pre_process)
    # displayimg(seg, 'bin')
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    get = getmatrix()
    # 在不同分割方式下，返回二维码bbox
    otsu_matrix_bbox = get.findcontours(otsu, (matrix_w_lower,matrix_w_upper,matrix_h_lower,matrix_h_upper))
    kmeans_matrix_bbox = get.findcontours(seg, (matrix_w_lower,matrix_w_upper,matrix_h_lower,matrix_h_upper))
    if (otsu_matrix_bbox is None) and (kmeans_matrix_bbox is None):
        logging.info("can't find matrix area")
        shutil.copy(img_path,os.path.join(abl_imgs, img_path.split('\\')[-1]))
        flag = 3
        return flag
    else:
        # 优先取信kmeans结果
        bbox = filtnone(kmeans_matrix_bbox,otsu_matrix_bbox)
        # 计算bbox Y方向最短边距
        # sobel_Y = calcsobel(pre_process, 3, 0, 1)
        # th, binary = cv2.threshold(sobel_Y,0,255,cv2.THRESH_OTSU)
        roof_dist = get_min_dist(seg, bbox)
        ground_dist = get_min_dist(seg, bbox,mode='ground')
        logging.info("roof_dist:{}, ground_dist:{}".format(roof_dist, ground_dist))
        if roof_dist<dist_thresh or ground_dist<dist_thresh:
            logging.info("matrix shifted over thresh!")
            shutil.copy(img_path,os.path.join(abl_imgs,img_path.split('\\')[-1]))
            flag = 3
            return flag
        # 剪裁解码
        cropped = pre_process[bbox[1]-extend_h:bbox[1]+bbox[3]+extend_h,
                              bbox[0]-extend_w:bbox[0]+bbox[2]+extend_w]
        cropped = cv2.resize(cropped,(150,150))
        matrix_bin = kmeans(cropped,n_cluster=None)
        if matrix_bin[0,0] != 0:
            matrix_bin = cv2.bitwise_not(matrix_bin)
        panelid = decode_datamatrix(matrix_bin, (k1,k1), (k2,k2))
        if panelid is not None:
            logging.info("%s decoding successfully"%img_path)
            logging.info("The panelid is %s"%panelid)
            return flag
        else:
            logging.info("%s decoding failed"%img_path)
            flag = 1
            shutil.copy(img_path,os.path.join(abl_imgs,img_path.split('\\')[-1]))
            return flag
        
if __name__ == '__main__':

    # if len(sys.argv) > 1:
    img_paths = glob.glob("data/test_img/*.jpg")
    for img_path in img_paths:
        start = time.time()
        result = main(img_path)
        # else:
        #     print('Usage: python main.py ImageFile')
        logging.info("process time: {:.2f}s".format(time.time()-start))

    




