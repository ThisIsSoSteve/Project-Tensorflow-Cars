import cv2
import numpy as np

def filter(image):

    lower_limit = np.array([0, 0, 0])
    upper_limit = np.array([150, 10, 160])

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_limit, upper_limit)

    result = cv2.bitwise_and(image, image, mask=mask)

    return result

    #cv2.imshow('image',image)
    #cv2.imshow('mask',mask)
    #cv2.imshow('result',result)
    #cv2.waitKey(0)

    #cv2.destroyAllWindows()


#image = cv2.imread('E:/repos/pics/colour14.png', 1)
#cv2.imshow('image',image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#filter(image)