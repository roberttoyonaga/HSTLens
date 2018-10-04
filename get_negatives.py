
import pandas as pd
from daomop.storage import tap_query
import os
import time 
import subprocess
from astropy import wcs
from astropy.io import fits
from astropy.nddata.utils import Cutout2D
import numpy as np
import random
from sklearn.preprocessing import minmax_scale

hm_lenses = input("How many lenses do you want to generate parameter files for? ")


def dwnld(entry, data_dict):
    subprocess.check_output(["wget", "-O","/run/media/toyonagar/Lexar/negatives/cutout_"+str(entry)+ ".fits" ,
                             "http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/caom2ops/cutout?uri=mast:HST/product/"+
                             data_dict["obj_name"][entry]+"_drz.fits&cutout=Circle+ICRS+" + str(data_dict["ra"][entry]) + "+" +
                             str(data_dict["dec"][entry]) + "+0.001"]) #tried 0.0005

def get_filenames(ra, dec, data_dict):
    query = '''SELECT caom2.Plane.publisherID, caom2.Plane.planeURI 
    FROM caom2.observation 
    JOIN caom2.plane ON caom2.observation.obsID = caom2.plane.obsID 
    WHERE CONTAINS( CIRCLE('ICRS',''' + str(ra) + ',' +str(dec) + ''',0.02), caom2.plane.position_bounds)=1 
    AND caom2.observation.collection = 'HST' 
    AND caom2.observation.instrument_name = 'ACS/WFC' '''

    out_query = tap_query(query)

    '''store the pubID and filenames'''
    filled = False
    for entry in range(len(out_query["publisherID"])):
        if ("0-" in out_query["publisherID"][entry]) and ("PRODUCT" in out_query["publisherID"][entry]):
            data_dict["pubID"].append(out_query["publisherID"][entry])

            start = out_query["publisherID"][entry].find('HST?') + 4 #add 4 to get to the and of the HST?
            end = out_query["publisherID"][entry].find('/j', start)
            data_dict["obj_name"].append(out_query["publisherID"][entry][start:end])
            filled=True
            break  #only add one entry per ADQL query. Otherwise you'll download different images of the same obj
    
    if filled == False: 
            data_dict["obj_name"].append(None)
    return data_dict





'''extract data from table'''

blank = '      '
star = 'STAR     '
unknown = 'UNKNOWN  '
galaxy ='GALAXY   '
dataframe = pd.read_csv("with_zphot.tsv",delimiter=";")

data_dict ={"ra": [], "dec":[], "pubID":[], "obj_name":[] } #each entry is a different row

for i in range(len(dataframe)):
    if (dataframe['zsp'][i]==blank) or (dataframe.Class[i]==star or dataframe.Class[i]=='EMISSION '):
        data_dict["ra"].append(dataframe["RAJ2000"][i])
        data_dict["dec"].append(dataframe['DEJ2000'][i])
        
        
        
        
'''tap query for the potential results'''
print("========== you can make this many lenses: "+str(len(data_dict['ra'])))
start=time.time()
for i in range(hm_lenses):#range(len(data_dict['ra'])):
    print(i)
    data_dict = get_filenames(data_dict['ra'][i], data_dict['dec'][i], data_dict)
    
print(time.time()-start) #took 435.431326866 for 561 queries. Will get repeated filenames, do not download repeats



'''download image cutouts based on the file names''' #needs error handling badly for when you cant get a 0- PRODUCT name!!
for i in range(len(data_dict["obj_name"])):
    print("dwnlding image "+str(i))
    if data_dict["obj_name"][i] != None:  #wnld only if you found a 0- PRODUCT image
        try: 
            dwnld(i, data_dict)
        except:
            print("Failed")

            
'''after downloading, find the pixel coordinates of the RA/dec that you have and then cut a 100/100 square around there '''
image_number = 0 #want to make sure NO numbers are missing. To make importing the finished data easier
for cutout in range(len(data_dict["obj_name"])):
    try:
        data, hdr = fits.getdata("/run/media/toyonagar/Lexar/negatives/cutout_"+str(cutout)+".fits", 1, header=True) 
        w = wcs.WCS(hdr)
        pixcrd2 = w.wcs_world2pix([[float(data_dict['ra'][cutout]), float(data_dict['dec'][cutout])]], 0)
        print (pixcrd2)
        centered_cut = Cutout2D(data, (pixcrd2[0][0], pixcrd2[0][1]), (100, 100))
        print (centered_cut.shape)
        
        #do a 0-1 scaling of the pixel intensities
        tmp_image = centered_cut.data.reshape(-1,)
        tmp_image = minmax_scale(tmp_image, feature_range = (0,1))
        centered_cut.data = tmp_image.reshape(centered_cut.data.shape[0],centered_cut.data.shape[1])
        
        #save the image
        fits.writeto('/run/media/toyonagar/Lexar/negatives/out'+str(image_number)+'.fits',
                     centered_cut.data, header=hdr, overwrite=True)
        image_number +=1
    except:
        print("skipped due to non-existent image, or conversion error")

        

