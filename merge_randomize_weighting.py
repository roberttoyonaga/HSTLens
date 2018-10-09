from astropy.nddata.utils import Cutout2D
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale

LENSES_PATH = "lenses/lens_image_"
OUT_PATH = "/run/media/toyonagar/Lexar/out/out"
MERGED_PATH = "/run/media/toyonagar/Lexar/merged/merged_"
hm_lenses = int(input("How many lenses do you want to merge with associated cutouts? "))

alpha = [0.85, 0.65, 0.45, 0.25]

def normalize( all_cutouts, all_lenses, hm_lenses): #only for 1 color channel. Scale both (0,1)
    bad_images = []
    for lens_number in range(hm_lenses):
        try:
            for i in range(len(alpha)): #NEED TO FIT 4 INTENSITIES INTO 1 ARRAY
                tmp_lens = all_lenses[lens_number][0].reshape(-1,)
                tmp_lens = minmax_scale(tmp_lens, feature_range = (0, 1*alpha[i]))
                all_lenses[lens_number+ len(all_lenses)*i][0] = tmp_lens.reshape(all_lenses.shape[2],all_lenses.shape[2])
            
            tmp_lens2 = all_cutouts[lens_number][0].reshape(-1,)
            tmp_lens2 = minmax_scale(tmp_lens2, feature_range = (0,1))
            all_cutouts[lens_number][0] = tmp_lens2.reshape(all_cutouts.shape[2],all_cutouts.shape[2])
    
        except:
            bad_images.append(lens_number)
            print ("image contains NaN")
    
    hm_lenses -=len(bad_images)        
    all_lenses = np.delete(all_lenses,bad_images, axis=0)
    all_cutouts = np.delete(all_cutouts,bad_images, axis=0)

    
 
    return all_cutouts, all_lenses
    

def load_images(sample_num): #NEED TO LOAD THE SAME IMAGE FOUR TIMES INTO all_cutouts
    
    try:
        lens = fits.open(LENSES_PATH+str(sample_num)+".fits")[0].data #PATH will  run dir
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
        fits.writeto(MERGED_PATH + str(image)+'.fits', merged[image], overwrite =True) 
        
def save_image(image_number, merged):
    fits.writeto(MERGED_PATH + str(image_number)+'.fits', merged[image_number], overwrite =True) 

def sum_images(all_cutouts, all_lenses): #only sums in 2D, no color channel,need to save periodically
    summed = []
    
    for image_number in range(hm_lenses):    
        try:
            summed.append(np.add(all_cutouts[image_number][0], all_lenses[image_number][0])) 
        
            save_image(image_number, summed[image_number])
        except:
            print("failed at addimg images together "+str(image_number))
            
    return summed


all_cutouts = np.zeros((hm_lenses*len(alpha),1,100,100))
all_lenses = np.zeros((hm_lenses*len(alpha),1,100,100))

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

all_cutouts, all_lenses = normalize(all_cutouts, all_lenses, hm_lenses)

print("images normalized")   
    
merged = sum_images(all_cutouts, all_lenses)  

print("images merged") 
'''
bad_images = []
for image in range(len(merged)):
    try:
        tmp_image = merged[image].reshape(-1,)
        tmp_image = minmax_scale(tmp_image, feature_range = (0,1))
        merged[image] = tmp_image.reshape(merged[image].shape[0],merged[image].shape[1])
        '''
        tmp_out = all_cutouts[image][0].reshape(-1,)
        tmp_out = minmax_scale(tmp_out, feature_range = (0,1))
        all_cutouts[image][0] = tmp_out.reshape(all_cutouts[image][0].shape[0],all_cutouts[image][0].shape[1])
        '''
    except:
        bad_images.append(image)
        print ("image contains NaN")
        
hm_lenses -=len(bad_images)        
merged = np.delete(merged,bad_images, axis=0)

print ("images re-normalized")
'''
        
#save_images(hm_lenses, merged)
#print(" merged images saved ")