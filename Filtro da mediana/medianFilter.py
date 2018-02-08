## Maria Júlia de Oliveira Vieira

import cv2
import numpy as np
from matplotlib import pyplot as plt

#display image
def showImage(img):
    cv2.imshow('cat',img)
    
    while True:
        k = cv2.waitKey(0) & 0xFF     
        if k == 27: break             # ESC key to exit 
    cv2.destroyAllWindows()

#Clean the noise, but the image gets too blurred
def blurBlur(img):
    blur = cv2.blur(img,(5,5))
    showImage(blur)

#Clean the noise, but the image gets too blurred
def medianBlur(img):
    median = cv2.medianBlur(img,5)
    showImage(median)

#Clean the noise, and the image still focus (Best blur (Y))
def gaussianBlur(img):
    blur = cv2.GaussianBlur(img,(5,5),0)
    showImage(blur)

#Clean the noise, but the image gets too blurred
def bilateralBlur(img):
    blur = cv2.bilateralFilter(img,9,75,75)
    showImage(blur)

def main():
    big_cat_img = cv2.imread('big_cat.png')
    #medianBlur(big_cat_img)
    #gaussianBlur(big_cat_img)
    #bilateralBlur(big_cat_img)
    blurBlur(big_cat_img)

if __name__=='__main__':
    main()