from astropy.nddata.utils import Cutout2D
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

OUT_PATH = "/run/media/toyonagar/Lexar/out/out"
MERGED_PATH = "/run/media/toyonagar/Lexar/merged/merged_"
def normalize( all_cutouts, all_lenses): #only for 1 color channel
    
    for lens_number in range(hm_lenses):
        tmp_lens = all_lenses[lens_number][0].reshape(-1,)
        tmp_lens = minmax_scale(tmp_lens, feature_range = (all_cutouts[lens_number][0].min(), all_cutouts[lens_number][0].max()*0.75))
        all_lenses[lens_number][0] = tmp_lens.reshape(all_lenses.shape[2],all_lenses.shape[2])
    '''
    minmax_cutout = make_scaler2(all_cutouts)
    minmax_lens = make_scaler2(all_lenses)
    all_cutouts = scale_data2(all_cutouts, minmax_cutout)
    all_lenses = scale_data2(all_lenses, minmax_lens)
    '''
    return all_cutouts, all_lenses
    

def load_images(sample_num):
    
    try:
        lens = fits.open("lens_image_"+str(sample_num)+".fits")[0].data #PATH will  run dir
    except:
        print "lens messed up"
    try:
        cutout = fits.open(OUT_PATH+str(sample_num)+".fits")[0].data 
    except:
        print "out messed up"
    try:
        cutout = cutout.reshape(1,cutout.shape[0],cutout.shape[1])
        lens = lens.reshape(1,lens.shape[0],lens.shape[1])
    except: 
        print "reshape" 
    return cutout, lens
    

def save_images(hm_lenses, merged):
    
    for image in range(len(merged)):
        fits.writeto(MERGED_PATH + str(image)+'.fits', merged[image], overwrite =True) #PATH
    

def sum_images(all_cutouts, all_lenses): #only sums in 2D, no color channel
    summed = []
    for image_number in range(hm_lenses):    
        summed.append(np.add(all_cutouts[image_number][0], all_lenses[image_number][0])) 
    return summed

hm_lenses = input("How many lenses do you want to merge with associated cutouts? ")

all_cutouts = np.zeros((hm_lenses,1,100,100))
all_lenses = np.zeros((hm_lenses,1,100,100))

bad_images = [] #keep track of and remove bad images later

for i in range(hm_lenses):
    try:
        all_cutouts[i], all_lenses[i] = load_images(i)
    except:
        bad_images.append(i)
        print("couldnt find image: " +str(i))
        
hm_lenses -=len(bad_images)        
all_cutouts = np.delete(all_cutouts,bad_images, axis=0)
all_lenses = np.delete(all_lenses,bad_images, axis=0)
    
print("images loaded") 

all_cutouts, all_lenses = normalize(all_cutouts, all_lenses)
print("images normalized")   
    
merged = sum_images(all_cutouts, all_lenses)  

print("images merged") 

bad_images = []
for image in range(len(merged)):
    try:
        tmp_image = merged[image].reshape(-1,)
        tmp_image = minmax_scale(tmp_image, feature_range = (0,1))
        merged[image] = tmp_image.reshape(merged[image].shape[0],merged[image].shape[1])
    except:
        bad_images.append(image)
        print ("image contains NaN")
hm_lenses -=len(bad_images)        
merged = np.delete(merged,bad_images, axis=0)

    
print ("images re-normalized")

save_images(hm_lenses, merged)

print(" merged images saved ")