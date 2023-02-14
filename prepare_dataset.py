
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

train_dir = 'E:/' 
output_dir = 'D:/SRGAN_FLicker images/Data'

for img in os.listdir( train_dir + "mirflickr"):
    img_array = cv2.imread(train_dir + "/mirflickr/" + img)
    
    img_array = cv2.resize(img_array, (128,128))
    lr_img_array = cv2.resize(img_array,(32,32))
    cv2.imwrite(output_dir+ "/hr_images/" + img, img_array)
    cv2.imwrite(output_dir+ "/lr_images/"+ img, lr_img_array)