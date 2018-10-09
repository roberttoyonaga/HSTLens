
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

CUTOUT_PATH = "/run/media/toyonagar/Lexar/cutout/cutout_"
OUT_PATH ='/run/media/toyonagar/Lexar/out/out'
LENSES_PATH = "lenses/lens_image_"
hm_lenses = input("How many lenses do you want to generate parameter files for? ")


def dwnld(entry, data_dict): 
    subprocess.check_output(["wget", "-O",CUTOUT_PATH +str(entry)+ ".fits" ,
                             "http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/caom2ops/cutout?uri=mast:HST/product/"+
                             data_dict["obj_name"][entry]+"_drz.fits&cutout=Circle+ICRS+" + str(data_dict["ra"][entry]) + "+" +
                             str(data_dict["dec"][entry]) + "+0.001"]) 

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


def image_params(lens):
    #Set the lensing source parameters in the files that lenstool requires, and output them as new files
 
    my_file = open("temp_parameters.parorig","r") #changed file extension so bash wont delete after each iteration
    lines = my_file.readlines()
    my_file.close()

    for line in range(len(lines)):
        if "SOURCEFLAG" in lines[line]:
            lines[line] = "\t" +"source     1 cusp_left_" +str(lens)+".cat" +"\n" #PATH tmp.cat files in run dir
        if "PIXELFLAG" in lines[line]:
            lines[line] =  "\t" +"pixel     1 100 "+LENSES_PATH+str(lens)+".fits" +"\n" #PATH exports to run dir/lenses/
        if "SKYFLAG" in lines[line]:
            lines[line] =  "\t" +"sky     "+str(sky[lens]) +"\n"
        if "ZFLAG" in lines[line]:
            lines[line] =  "\t" +"z_lens     "+str(zlens[lens]) +"\n"
        if "ELLIPFLAG" in lines[line]:
            lines[line] =  "\t" +"ellipticite     "+str(ellip[lens]) +"\n"
        if "XCENTFLAG" in lines[line]:
            lines[line] =  "\t" +"x_centre     "+str(x_cent[lens]) +"\n"
        if "YCENTFLAG" in lines[line]:
            lines[line] =  "\t" +"y_centre     "+str(y_cent[lens]) +"\n"
        if "RADFLAG" in lines[line]:
            lines[line] =  "\t" +"core_radius_kpc     "+str(core_rad_kpc[lens]) +"\n"

    f = open("temp_parameters_"+str(lens)+".par","w") #PATH temp .par files in run dir
    f.writelines(lines)
    f.close()

    
def source_params(lens):
    #set up the parameters of the lensed source
    
    source_lines = []
    source_lines.append("#REFERENCE 0 \n")
    source_lines.append("S1 "+str(ra[lens])+ " "+str(dec[lens])+" "+str(a_maj[lens])+ " "+ str(b_maj[lens])+ " "+ str(theta[lens])+ " "+str(redshift[lens])+ " "+str(magnitude[lens]))
    f = open("cusp_left_" +str(lens)+".cat","w") #PATH leave as run dir
    f.writelines(source_lines)
    f.close()



'''extract data from table'''

blank = '      '
star = 'STAR     '
unknown = 'UNKNOWN  '
galaxy ='GALAXY   '
dataframe = pd.read_csv("with_zphot.tsv",delimiter=";")

data_dict = {"ra": [], "dec":[], "z_spect":[], "z_phot": [], "pubID":[], "obj_name":[] } 

for i in range(len(dataframe)):
    if (dataframe['zsp'][i]!=blank or dataframe.zbest[i]!=blank) and (dataframe.Class[i]==galaxy or dataframe.Class[i]==unknown):
        data_dict["ra"].append(dataframe["RAJ2000"][i])
        data_dict["dec"].append(dataframe['DEJ2000'][i])
        data_dict["z_spect"].append(dataframe["zsp"][i])
        data_dict["z_phot"].append(dataframe['zphot'][i])
        
        
        
'''tap query for the potential results'''
print("========== you can make this many lenses: "+str(len(data_dict['ra'])))
start=time.time()
for i in range(hm_lenses):#range(len(data_dict['ra'])):
    print(i)
    data_dict = get_filenames(data_dict['ra'][i], data_dict['dec'][i], data_dict)
    
print(time.time()-start)



'''download image cutouts based on the file names''' #needs error handling badly for when you cant get a 0- PRODUCT name!!
fails=0
for i in range(len(data_dict["obj_name"])):
    print("dwnlding image " + str(i))
    if data_dict["obj_name"][i] != None:  #wnld only if you found a 0- PRODUCT image
        try: 
            dwnld(i, data_dict)
        except:
            fails+=1
            print("Failed")
print("failed: "+str(fails))
            
'''after downloading, find the pixel coordinates of the RA/dec that you have and then cut a 100/100 square around there '''
for cutout in range(len(data_dict["obj_name"])):
    try:                                
        data, hdr = fits.getdata(CUTOUT_PATH + str(cutout) + ".fits", 1, header=True) #'sci' image[1] data and header
        w = wcs.WCS(hdr)
        pixcrd2 = w.wcs_world2pix([[float(data_dict['ra'][cutout]), float(data_dict['dec'][cutout])]], 0)
        print (pixcrd2)
        centered_cut = Cutout2D(data, (pixcrd2[0][0], pixcrd2[0][1]), (100, 100))
        print (centered_cut.shape)
        fits.writeto(OUT_PATH + str(cutout) + '.fits', centered_cut.data, header=hdr, overwrite =True)
    except:
        print("skipped due to non-existent image, or conversion error")

        
        
'''Generate the lenses based on the characteristics of each cutout stored in data_dict'''
'''
requires temp_parameters.par in same directory because this file contains the flags that this script searches for.

Will output 2 files: 
    temp_parameters_<lens number>.par  contains lensing source param
    cusp_left_<lens number>.cat       contains lensed source param

To generate the actual .fits files, define path to lenstool executable, provide name of par file :
    /home/toyonagar/anaconda3/HST_Lens/lenstool-7.1-linux64/lenstool ./temp_parameters_<lens number>.par -n 

    The lens will be generated in a file called lens_image_<lens number>.fits int he same directory
'''


print("======== make_lenses.py entered ==============")
#hm_lenses = input("How many lenses do you want to generate parameter files for? ")
zlens = []
ellip = []
sky = []
x_cent = []
y_cent = []
core_rad_kpc =[]
ra = []
dec = []
a_maj = []
b_maj = []
theta = []
redshift = []
magnitude = [] 


    
#Here we randomize the lensing and lensed source parameters   
for i in range(hm_lenses):   
    if data_dict['z_spect'][i] != blank:    
        zlens.append(data_dict['z_spect'][i]) #spectroscopic redshift is better
    else:
        zlens.append(data_dict['z_phot'][i])
    
    sky.append(1) #set to 1 for no neg values
    ellip.append(float(random.randint(0,90))/100.)
    x_cent.append(float(random.randint(0,100))/1000.) 
    y_cent.append(float(random.randint(0,100))/1000.)
    core_rad_kpc.append(float(random.randint(400,700))/100.) #set to btw 10 and 14. Smaller makes it larger
    
    ra.append(float(random.randint(0,70))/100000.)
    dec.append(-float(random.randint(0,70))/100000.)
    a_maj.append(float(random.randint(0,100))/100.)
    b_maj.append(float(random.randint(0,100))/100.)
    theta.append(float(random.randint(0,100))/100.)
    
    redshift.append( float(zlens[i])+ float(random.randint(0,100))/100.) #lensed source must have more redshift than the lensing source
    while(redshift[i]>8):
        print("trying to get source redshift under 8")
        redshift[i] = float(zlens[i])+ float(random.randint(5,70))/100. #larger makes smaller rings
    
    magnitude.append(float(random.randint(120,220))/10.)
    
    print("===== parameter randomizing done ======")

    
#Now actually generate the parameter files for lenstool based on the randomized values
for lens in range(hm_lenses):  
    try: 
        image_params(lens)
        source_params(lens)
        print("making files for lens "+str(lens))
    except: 
        print("Skipping. cutout_ "+str(lens)+" doesnt exist")
    
    

