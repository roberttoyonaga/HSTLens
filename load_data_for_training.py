#import all the data. This will load the data into "ims" (set_size, color, 100,100) and "y" (set_size , 1) 
import numpy as np
import os 
from astropy.io import fits

NEG_PATH = "/home/roberttoyonaga/notebooks/roberttoyonaga/CMUDeepLensNRC/NRC_DeepLens/dataset_sim/negatives/"
POS_PATH = "/home/roberttoyonaga/notebooks/roberttoyonaga/CMUDeepLensNRC/NRC_DeepLens/dataset_sim/merged/"

set_size =1000

ims = np.ones((set_size, 1, 100, 100))
y = np.ones((set_size,1))

#load the negatives into even numbered indices (including 0)
count =0
for filename in os.listdir(NEG_PATH):
    if filename.endswith(".fits") and count<set_size/2: 
        ims[count*2] = fits.open(NEG_PATH+filename)[0].data
        y[count*2] = np.array([0])
        count+=1
    
    
#load the positives into odd numered indices
count =0
for filename in os.listdir(POS_PATH):
    if filename.endswith(".fits") and count<set_size/2: 
        ims[count*2+1] = fits.open(POS_PATH+filename)[0].data
        y[count*2+1] = np.array([1])
        count+=1
    