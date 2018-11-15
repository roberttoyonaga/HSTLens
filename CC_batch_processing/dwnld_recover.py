import pandas as pd
import subprocess
import os

begin = int(input("starting point"))
jobs = pd.read_csv('/home/toyonaga/projects/def-sfabbro/toyonaga/HSTLens/CC_batch_processing/missed_jobs.csv')
start_point= jobs['job'][begin]

batch_size = 1
DWNLD_PATH = "/home/toyonaga/scratch/pipeline/full/full_"+str(start_point)
CSV_PATH="test_csv" #shouldn't have to change

im_info = pd.read_csv(CSV_PATH)

def dwnld(entry): 
    subprocess.check_output(["wget", "-O",DWNLD_PATH+ ".fits" ,
                             "http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/MAST/HST/product/"+
                             im_info['obj_name'][entry]+"_drz.fits"]) 
    
'''download image cutouts based on the file names'''
fails=0
for i in range(batch_size):
    try: 
        dwnld(i+int(start_point))
    except:
        fails+=1
        print("Failed")
print("failed: "+str(fails))

os.system('sex /home/toyonaga/scratch/pipeline/full/full_'+ str(start_point)+'.fits -c parameters.se -CATALOG_NAME /home/toyonaga/scratch/pipeline/cat/image_'+str(start_point)+'.cat ')