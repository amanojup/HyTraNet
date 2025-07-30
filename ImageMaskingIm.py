# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 15:50:18 2021

@author: DELL
"""

# Required package
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from time import sleep
import cv2
import os
from numpy import asarray
import datetime
def current_time():
    return datetime.datetime.now().strftime("%H_%M_%S_%d_%m_%Y")
# Boundary value Defined 
lower_b = np.array([0,0,0]).astype('uint8')
upper_b = np.array([187,81,190]).astype('uint8')

lower_r = np.array([0,0,160]).astype('uint8')
upper_r = np.array([120,108,255]).astype('uint8')

lower_g = np.array([0,200,0]).astype('uint8')
upper_g = np.array([200,225,180]).astype('uint8')

lower_o = np.array([0,0,250]).astype('uint8')
upper_o = np.array([165,190,255]).astype('uint8')

# Read image using opencv module 
input_img = cv2.imread('D:\8_Article2021_ImageBased\ImageBasedPaper\GoogleImage\Delhi_17_49_21_06_03_2022_PresultsO.png')

#input_img = cv2.imread(os.path.join(data_path, input_img))

#Raw image converted to get binary image
#All the pixel value in the range have value 255 and other have 0
mask_b = cv2.inRange(input_img, lower_b, upper_b)
mask_r = cv2.inRange(input_img, lower_r, upper_r)
mask_g = cv2.inRange(input_img, lower_g, upper_g)
mask_o = cv2.inRange(input_img, lower_o, upper_o)
#Bitwise multiplication is done between input image and masking image
#resulting in image having only specific color 
print(mask_o.shape)
print(max(mask_o[0]))

output_b = cv2.bitwise_and(input_img, input_img, mask=mask_b)
output_r = cv2.bitwise_and(input_img, input_img, mask=mask_r)
output_g = cv2.bitwise_and(input_img, input_img, mask=mask_g)
output_o = cv2.bitwise_and(input_img, input_img, mask=mask_o)
#add all the individual color into one
combine_im1 = cv2.add(output_b, output_r)
output_img = cv2.add(combine_im1, output_g)
output_RGO = cv2.add(output_img, output_o)
#Show the image using Opencv function.
im1 = cv2.imshow('o', output_RGO)
cv2.waitKey(0)
#pp = asarray(im1)
#cv2.imwrite("D:\8_Article2021_ImageBased\segmentation--master\prediction_paper\prediction_paper\masked1\Delhi_"+current_time()+".png",output_RGO)
    #sleep(1)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    #print('Data Type: %s' % pp.dtype)