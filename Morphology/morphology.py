'''

- operation (dilation, erosion, opening, closing).
- wsize (0, 1, 3, ..., 21).
- kernel (Square, ellipse, cross).
'''


import cv2 as cv

def call(x):
    pass

windowName = 'Morphology'
kSizeName = 'Kernel Size'
operationName = 'Operator:\n 0: Dilate - 1: Erode \n 2: Opening - 3: Closing'
kernelName = 'Kernel:\n 0: Square - 1: Cross - 2: Ellipse'

img  = cv.imread('img1.jpeg',0)
imgC = img.copy()

cv.namedWindow(windowName)
#Create Trackbar to choose kernel size
cv.createTrackbar(kSizeName, windowName, 1, 21, call)
#Create Trackbar to select Morphology operation
cv.createTrackbar(operationName, windowName, 1, 4, call)
#Create Trackbar to select kernel type
cv.createTrackbar(kernelName, windowName, 1, 3, call)

while(True):

    cv.imshow(windowName, imgC)

    ksize = cv.getTrackbarPos(kSizeName, windowName)
    op = cv.getTrackbarPos(operationName, windowName)
    ker = cv.getTrackbarPos(kernelName, windowName)

    if ksize == 0:
        ksize = 1

    if ker == 0:
        kernel = cv.getStructuringElement(cv.MORPH_RECT,(ksize,ksize))
    elif ker == 1:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(ksize,ksize))
    else:
        kernel =  cv.getStructuringElement(cv.MORPH_CROSS,(ksize,ksize))
    
    if op == 0:
        imgC = cv.erode(img,kernel,iterations = 1)
    elif op == 1:
        imgC = cv.dilate(img,kernel,iterations = 1)
    elif op == 2:
        imgC = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    else:
        imgC = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)
    
    q = cv.waitKey(0) 
    if q == 'q': break

cv.destroyAllWindows()