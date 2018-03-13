import cv2
import numpy as np

def callback(img):
    pass

def createWindow(img):
    wind = "Smoothing"
    slider_name = "1-Aver 2-Med 3-Gau 4-Bil"
    slider_Pos = 0
    image = img.copy()
    #create window
    cv2.namedWindow(wind)
    #show image
    cv2.imshow(wind, image)
    #add slider (slider_name, window_name, start_value, max_value, callback)
    cv2.createTrackbar(slider_name, wind, 1, 4, callback)
    while(cv2.waitKey(1000)):
        cv2.imshow(wind, image)
        slider_Pos = cv2.getTrackbarPos(slider_name, wind)
        if(slider_Pos == 1):
            image = averageBlur(img)
        elif(slider_Pos == 2):
            image = meidanBlur(img)
        elif(slider_Pos == 3):
            image = gaussianBlur(img)
        else:
            image = bilateralFilt(img)
        
    cv2.destroyAllWindows()
    

# Takes the average of all the pixels under kernel area and replace the central element.
def averageBlur(img):
    return cv2.blur(img,(5,5))

# Takes median of all the pixels under kernel area and central element is replaced with this median value.
def meidanBlur(img):
    return cv2.medianBlur(img,5) 

# Gaussian blurring is highly effective in removing gaussian noise from the image.
#This gaussian filter is a function of space alone, that is, nearby pixels are considered while filtering.
def gaussianBlur(img):
    return cv2.GaussianBlur(img,(5,5),0) 

# Takes a gaussian filter in space, but one more gaussian filter which is a function of pixel difference.
def bilateralFilt(img):
    return cv2.bilateralFilter(img,9,75,75) 

def main():
    kitty = cv2.imread('big_cat.png')
    createWindow(kitty)
    

if __name__=='__main__':
    main()
       