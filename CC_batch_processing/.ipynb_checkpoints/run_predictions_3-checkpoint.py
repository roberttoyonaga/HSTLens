from astropy.io import fits
import astropy.wcs as wcs
import numpy as np
from astropy.nddata.utils import Cutout2D
from sklearn.preprocessing import minmax_scale
import time
import tensorflow
import keras
from keras.layers import Activation
from keras.layers import Conv2D, ELU
from HSTLens_resnet16_3_classes import deeplens_classifier
import pandas as pd
import os

start_point = input("start_point")
IMAGE_PATH = '/home/toyonaga/scratch/pipeline/full/full_'+str(start_point)
width=100
mini_batch_sz =500
hm_models = 3. #how many estimators
OUT_PATH= "/home/toyonaga/scratch/pipeline/results/results_"+str(start_point)+".out"
SE_PATH = "/home/toyonaga/scratch/pipeline/cat/image_"+str(start_point)+".cat"
CSV_PATH="test_csv" #shouldn't  have to change
WEIGHTS_PATH = "/home/toyonaga/projects/def-sfabbro/toyonaga/HSTLens/CC_batch_processing/batchjob3_weigths/" #ensembleweights_"

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

    
    
'''Get the ra/dec and make cutouts'''
'''for each detected obj in the large image, make a cutout and calculate/store the ra/dec'''
start= time.time()
crd_dict = {'obj_name':[],'ra': [], 'dec': [], 'prb_1':[],'prb_2':[],'prb_3':[]}
im_info = pd.read_csv(CSV_PATH)


hm_batches=len(xl)//int(mini_batch_sz)
last_batch = len(xl)%mini_batch_sz

data, hdr = fits.getdata(IMAGE_PATH + ".fits", 0, header=True) #load image into memory only once  (18s vs. 0.28s)
data = np.nan_to_num(data)
w = wcs.WCS(hdr)
                
predictions = [[],[],[]]#estimator[predictions[class]]

weight_names = os.listdir(WEIGHTS_PATH)

my_models = []
for model in range(int(hm_models)):
    my_models.append(deeplens_classifier())
    my_models[model]._build() #should not accept arguments
    my_models[model].model.load_weights(WEIGHTS_PATH+weight_names[model])

'''for each detected obj in the large image, make a cutout and calculate/store the ra/dec'''
for batch in range(hm_batches+1): #for 500 batches
    curr = batch*mini_batch_sz #get the start of the current mini batch
    bad_images = []

    ims = np.zeros((mini_batch_sz,1, width, width))
    if batch ==hm_batches+1-1:
        ims = np.zeros((last_batch,1, width, width))
    for cutout in range(len(ims)): #prepare each image in 500
        try:
            centered_cut = Cutout2D(data,(float(xl[cutout+curr]), 
                                          float(yl[cutout+curr])), (width, width)).data # +random.randint(-15,15)

            centered_cut = np.nan_to_num(centered_cut)
      
            if centered_cut.all() ==0:    #check for all Nan images 
                bad_images.append(cutout)
                continue
                
            if centered_cut.shape != (100, 100):    #check for cut off images
                result = np.zeros((100,100))
                result[:centered_cut.shape[0],:centered_cut.shape[1]] = centered_cut
                centered_cut = result
            
            #normalize
            tmp_image = centered_cut.reshape(-1,)                        #scale       
            tmp_image = minmax_scale(tmp_image, feature_range = (0,1))
            centered_cut = tmp_image.reshape(centered_cut.shape[1],centered_cut.shape[0]) #images should be 100x100
            ims[cutout][0] = centered_cut

            #once the images is confirmed fine, then add to crd_dict
            worldcrd = w.all_pix2world(float(xl[cutout+curr]), float(yl[cutout+curr]),1) #1 for fits
            crd_dict['ra'].append(float(worldcrd[0]))
            crd_dict['dec'].append(float(worldcrd[1]))
            crd_dict['prb_1'].append(float(-1.))
            crd_dict['prb_2'].append(float(-1.))
            crd_dict['prb_3'].append(float(-1.))
            crd_dict['obj_name'].append(im_info['obj_name'][int(start_point)]+"_drz.fits")

        except:
            print("skipped due to non-existent image, or conversion error")
            bad_images.append(cutout)

    ims = np.delete(ims,bad_images, axis=0)
    print ("runtime: "+str(time.time()-start))       


    '''Get predictions on the 500 images in this batch using each of the 3 models'''
    for i in range(len(my_models)):
        predictions[i].append(my_models[i]._predict( ims, y= None, discrete = False))

print (np.array(predictions).shape)
for i in range(len(predictions)): #combine separate prediction arrays
    predictions[i] =np.concatenate(predictions[i]) #a 2D array of shape(len, #classes)
    
out = pd.DataFrame(crd_dict)
print("out csv len "+str(len(out))+' predictions len' +str(len(predictions))+' predictions[0].shape '+str(predictions[0].shape))

'''For each image, average the results of the 3 estimators (for each of the3 classes)'''

for prediction in range(len(predictions[0])):
    
    out['prb_1'][prediction] = np.average(np.array([predictions[0][prediction][0], predictions[1][prediction][0], predictions[2][prediction][0]]))
    
    out['prb_2'][prediction] = np.average(np.array([predictions[0][prediction][1], predictions[1][prediction][1], predictions[2][prediction][1]]))
    
    out['prb_3'][prediction] = np.average(np.array([predictions[0][prediction][2], predictions[1][prediction][2], predictions[2][prediction][2]]))

    
print('saving csv')
out.to_csv(OUT_PATH, index = False) #dont save the indexing


    
# try:
#     os.system('rm /home/toyonaga/scratch/pipeline/full/full_'
#              +str(start_point)+'.fits /home/toyonaga/scratch/pipeline/cat/image_'
#              +str(start_point)+'.cat')
#     print("REMOVAL SUCCESSFUL")
    
# except:
#     print('removal of images failed')