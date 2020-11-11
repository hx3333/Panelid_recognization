# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 15:35:49 2020

@author: sshss
"""



import argparse

import cv2, os

def variance_of_laplacian(image):

  # compute the Laplacian of the image and then return the focus

  # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

# construct the argument parse and parse the arguments
def arg_parse():
    """
    Parse arguements to the detect module
    
    """

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--images", dest = 'images', 
                        default = "./blur_samples", type = str)
    parser.add_argument("--threshold", dest = 'threshold', 
                        default = 10, type = float)
        
    return parser.parse_args()


args = arg_parse()

for imagePath in os.listdir(args.images):

  # load the image, convert it to grayscale, and compute the

  # focus measure of the image using the Variance of Laplacian

  # method
    cut_edge = 3
    
    image = cv2.imread(os.path.join(args.images,imagePath))
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    gray = gray[cut_edge:-cut_edge,cut_edge:-cut_edge]
    
    fm = variance_of_laplacian(gray)
    
    text = "Not Blurry"
    
    # if the focus measure is less than the supplied threshold,
    
    # then the image should be considered "blurry"
    
    if fm < args.threshold:
    
      text = "Blurry"
    
    # show the image
    
    cv2.putText(gray, "{}: {:.2f}".format(text, fm), (10, 30),
    
      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    
    cv2.imshow("Image", gray)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()