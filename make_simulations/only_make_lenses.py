
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
hm_lenses = input("How many lenses do you want to generate parameter files for? ")

LENSES_PATH = "lenses/l_"
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

    zlens.append(float(random.randint(50,100))/100.) #smaller is bigger
    sky.append(1) #set to 1 for no neg values
    ellip.append(float(random.randint(0,50))/100.)
    x_cent.append(float(random.randint(0,200))/1000.) #200 #100 oct22
    y_cent.append(float(random.randint(0,200))/1000.)#200 #100oct22
    core_rad_kpc.append(float(random.randint(400,700))/100.) #400-700oct22. Smaller makes it larger
    
    ra.append(float(random.randint(0,35))/100000.)#closer to 0 the more circular
    dec.append(-float(random.randint(0,35))/100000.)#70 #50 oct 22
    a_maj.append(float(random.randint(0,100))/100.)
    b_maj.append(float(random.randint(0,100))/100.)
    theta.append(float(random.randint(0,100))/100.)
    
    redshift.append( float(zlens[i])+ float(random.randint(15,50))/100.) #30-80 oct22
    count =0
    while(redshift[i]>6 and count<10):
        count+=1
        print("trying to get source redshift under 8")
        redshift[i] = float(zlens[i])+ float(random.randint(10,20))/100. #larger makes smaller rings
    
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
    
    

