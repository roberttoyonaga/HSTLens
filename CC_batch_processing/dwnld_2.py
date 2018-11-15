import pandas as pd
import subprocess
batch_size = 2
start_point = input("starting point")

CSV_PATH="test_csv" #shouldn't have to change


im_info = pd.read_csv(CSV_PATH)

def dwnld(entry): 
    subprocess.check_output(["wget", "-O",DWNLD_PATH+ ".fits" ,
                             "http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/MAST/HST/product/"+
                             im_info['obj_name'][entry]+"_drz.fits"]) 
    
'''download image cutouts based on the file names'''
fails=0
for i in range(batch_size):
    DWNLD_PATH = "/home/toyonaga/scratch/pipeline/full/full_"+str( i +int(start_point))
    try: 
        dwnld(i+int(start_point))
    except:
        fails+=1
        print("Failed")
print("failed: "+str(fails))

