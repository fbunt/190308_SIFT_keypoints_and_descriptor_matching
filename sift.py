import cv2
import numpy as np
import matplotlib.pyplot as plt

I_1 = plt.imread('I_1.jpg')
I_2 = plt.imread('I_2.jpg')

sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.04,edgeThreshold=10,sigma=1.6)
kp1,des1 = sift.detectAndCompute(I_1,None)
kp2,des2 = sift.detectAndCompute(I_2,None)
