from astropy.io import fits
import astropy.wcs as wcs
import numpy as np
from astropy.nddata.utils import Cutout2D
from sklearn.preprocessing import minmax_scale
import time
import pandas as pd
import os

start_point = input("start_point")
IMAGE_PATH = "/home/robbie/repos/HSTLens/CC_batch_processing/full/full_"+str(start_point)
width=100
OUT_PATH = "/home/robbie/repos/HSTLens/CC_batch_processing/out/out"+str(start_point)
SE_PATH = '/home/robbie/repos/HSTLens/CC_batch_processing/cat/image_'+str(start_point)+'.cat'
'''Get the pixel coordinates from SE'''

count=0
xl = []
yl =[]

f = open(SE_PATH,'r')
while (True):
    try:
        text = f.readline()
        if count>2:                     #change according to SE parameters
            words = text.split()
            xl.append(words[0])
            yl.append(words[1])
#             print(words[0], words[1], words[2]) ##debug
    except:
        break #breakout once the end is reached
    count+=1


data, hdr = fits.getdata(IMAGE_PATH + ".fits", 0, header=True) #load image into memory only once  (18s vs. 0.28s)
data = np.nan_to_num(data)
w = wcs.WCS(hdr)


for cutout in range(len(xl)):
    try:
        centered_cut = Cutout2D(data,(float(xl[cutout]), 
                                      float(yl[cutout])), (width, width),wcs=w) # +random.randint(-15,15)

        centered_cut.data = np.nan_to_num(centered_cut.data)

        if centered_cut.data.all() ==0:    #check for all Nan images 
            bad_images.append(cutout)
            continue

        if centered_cut.data.shape != (100, 100):    #check for cut off images
            result = np.zeros((100,100))
            result[:centered_cut.data.shape[0],:centered_cut.data.shape[1]] = centered_cut.data
            centered_cut.data = result
        
        
        hdu1 = fits.PrimaryHDU(data=centered_cut.data, header=centered_cut.wcs.to_header())
        hdu1.writeto(OUT_PATH+"_"+str(cutout)+'.fits', overwrite =True) 

    except:
        print("skipped due to non-existent image, or conversion error")



    
try:
    os.system('rm /home/robbie/repos/HSTLens/CC_batch_processing/full/full_'+str(start_point)+'.fits /home/robbie/repos/HSTLens/CC_batch_processing/cat/image_'
             +str(start_point)+'.cat')
    print("REMOVAL SUCCESSFUL")
    
except:
    print('removal of images failed')