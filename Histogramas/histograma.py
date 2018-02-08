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

#Create and plot grayScale image histogram
def grayHist(img):
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    #Plot Histogram
    plt.hist(img.ravel(),256,[0,256])
    plt.show()

#Create and plot color histogram
def colorHist(img):
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()

#Add uniform noise to image
def uniformNoise(img):
    low = (0, 0, 0)
    high = (40, 40, 40)
    noise = img.copy()
    cv2.randu(noise, low, high)
    img_noise = noise + img
    colorHist(img_noise)

#Add normal noise to image
def normalNoise(img):
    low = (0, 0, 0)
    sigma = (40, 40, 40)
    noise = img.copy()
    cv2.randn(noise, low, sigma)
    img_noise = noise + img
    colorHist(img_noise)

def main():
    colorCat_img = cv2.imread('color-kitty.jpg')
    grayCat_img = cv2.imread('color-kitty.jpg', cv2.IMREAD_GRAYSCALE)
    #grayHist(grayCat_img)
    #colorHist(colorCat_img)
    #uniformNoise(colorCat_img)
    #normalNoise(colorCat_img)

if __name__ == '__main__':
    main()