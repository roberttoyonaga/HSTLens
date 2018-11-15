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
from HSTLens_resnet2 import deeplens_classifier
import pandas as pd


start_point = input("start_point")
IMAGE_PATH = '/home/toyonaga/scratch/pipeline/full/full_'+str(start_point)
width=100
mini_batch_sz =500
OUT_PATH= "/home/toyonaga/scratch/pipeline/results/results_"+str(start_point)+".out"
SE_PATH = "/home/toyonaga/scratch/pipeline/cat/image_"+str(start_point)+".cat"
CSV_PATH="test_csv" #shouldn't  have to change
WEIGHTS_PATH = "/home/toyonaga/projects/def-sfabbro/toyonaga/HSTLens/weights/combined_nonsubtracted_weights_resnet2_h2_55000"


'''Get the pixel coordinates from SE'''

count=0
xl = []
yl =[]

f = open(SE_PATH,'r')
while (True):
    try:
        text = f.readline()
        if count>16:
            words = text.split()
            xl.append(words[1])
            yl.append(words[2])
    except:
        break #breakout once the end is reached
    count+=1

    
    
'''Get the ra/dec and make cutouts'''
'''for each detected obj in the large image, make a cutout and calculate/store the ra/dec'''
start= time.time()
crd_dict = {'obj_name':[],'ra': [], 'dec': [], 'prb':[]}
im_info = pd.read_csv(CSV_PATH)


hm_batches=len(xl)//int(mini_batch_sz)
last_batch = len(xl)%mini_batch_sz

data, hdr = fits.getdata(IMAGE_PATH + ".fits", 0, header=True) #load image into memory only once  (18s vs. 0.28s)
data = np.nan_to_num(data)
w = wcs.WCS(hdr)
                
predictions = []
my_model = deeplens_classifier()
my_model._build() #should not accept arguments
my_model.model.load_weights(WEIGHTS_PATH)
'''for each detected obj in the large image, make a cutout and calculate/store the ra/dec'''
for batch in range(hm_batches+1):
    curr = batch*mini_batch_sz #get the start of the current mini batch
    bad_images = []
    c_sig=10
    ims = np.zeros((mini_batch_sz,1, width, width))
    if batch ==hm_batches+1-1:
        ims = np.zeros((last_batch,1, width, width))
    for cutout in range(len(ims)):
        try:
            print(xl[cutout+curr],yl[cutout+curr])
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
            c = np.nan_to_num(centered_cut)
            sky_c =3.*np.median(centered_cut)- 2.*np.mean(centered_cut) #calc sky
            sigma_c= np.std(c)          #calc sigma
            c[np.isnan(c)]=sky_c          #get rid of NAn
            c[c>(sky_c+c_sig*sigma_c)]=sky_c+c_sig*sigma_c    #clip values
            c= c-sky_c                                         #subtract sky
            tmp_lens2 = c.reshape(-1,)                        #scale 
            tmp_lens2 = minmax_scale(tmp_lens2, feature_range = (0,1))
            centered_cut = tmp_lens2.reshape(centered_cut.shape[1],centered_cut.shape[1])

            summed = centered_cut
            np.random.seed()
            h1 = (sky_c)*np.random.randn(10000)
            np.random.shuffle(h1)
            summed = np.add(summed,h1.reshape(100,100)) # you can add constants too
            tmp_image = summed.reshape(-1,)
            tmp_image = minmax_scale(tmp_image, feature_range = (0,1))
            summed = tmp_image.reshape(summed.shape[0],summed.shape[1])

            ims[cutout][0] = summed

            #once the images is confirmed fine, then add to crd_dict
            worldcrd = w.all_pix2world(float(xl[cutout+curr]), float(yl[cutout+curr]),1) #1 for fits
            crd_dict['ra'].append(float(worldcrd[0]))
            crd_dict['dec'].append(float(worldcrd[1]))
            crd_dict['prb'].append(float(-1.))
            crd_dict['obj_name'].append(im_info['obj_name'][int(start_point)]+"_drz.fits")

        except:
            print("skipped due to non-existent image, or conversion error")
            bad_images.append(cutout)

    ims = np.delete(ims,bad_images, axis=0)
    print ("runtime: "+str(time.time()-start))       


    '''Feed cutouts to model and make CSV with lens ra/dec'''
    predictions.append(my_model._predict( ims, y= None, discrete = False))

    

predictions =np.concatenate(predictions)
print(len(predictions))        
out = pd.DataFrame(crd_dict)
print("out csv len"+str(len(out))+'predictions len' +str(len(predictions)))
lens = []
neg= []
for prediction in range(len(predictions)):
    if (predictions[prediction] >0.5):
        lens.append(prediction)
        out['prb'][prediction] = predictions[prediction] 
    else:
        neg.append(prediction)
out = out.drop(out.index[neg]) #drop all the rows that are not lenses
out.to_csv(OUT_PATH, index = False) #dont save the indexing


    
