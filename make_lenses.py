'''
requires temp_parameters.par in same directory because this file contains the flags that this script searches for.

Will output 2 files: 
    temp_parameters_<lens number>.par  contains lensing source param
    cusp_left_<lens number>.cat       contains lensed source param

To generate the actual .fits files, define path to lenstool executable, provide name of par file :
    /home/toyonagar/anaconda3/HST_Lens/lenstool-7.1-linux64/lenstool ./temp_parameters_<lens number>.par -n 

    The lens will be generated in a file called lens_image_<lens number>.fits int he same directory
'''
import random

print("======== make_lenses.py entered ==============")
hm_lenses = input("How many lenses do you want to generate parameter files for? ")
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


def image_params(lens):
    #Set the lensing source parameters in the files that lenstool requires, and output them as new files
 
    my_file = open("temp_parameters.parorig","r") #changed file extension so bash wont delete after each iteration
    lines = my_file.readlines()
    my_file.close()

    for line in range(len(lines)):
        if "SOURCEFLAG" in lines[line]:
            lines[line] = "\t" +"source     1 cusp_left_" +str(lens)+".cat" +"\n"
        if "PIXELFLAG" in lines[line]:
            lines[line] =  "\t" +"pixel     1 200 lens_image_"+str(lens)+".fits" +"\n"
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

    f = open("temp_parameters_"+str(lens)+".par","w")
    f.writelines(lines)
    f.close()

def source_params(lens):
    #set up the parameters of the lensed source
    
    source_lines = []
    source_lines.append("#REFERENCE 0 \n")
    source_lines.append("S1 "+str(ra[lens])+ " "+str(dec[lens])+" "+str(a_maj[lens])+ " "+ str(b_maj[lens])+ " "+ str(theta[lens])+ " "+str(redshift[lens])+ " "+str(magnitude[lens]))
    f = open("cusp_left_" +str(lens)+".cat","w")
    f.writelines(source_lines)
    f.close()
    
    
for i in range(hm_lenses):   
    #Here we randomize the lensing and lensed source parameters
    
    zlens.append( float(random.randint(1,30))/10.) #rand btw 0.1 and 3 precise to 1 dec places
    sky.append(100)
    ellip.append(float(random.randint(0,100))/100.)
    x_cent.append(float(random.randint(1,500))/10000.) 
    y_cent.append(float(random.randint(0,500))/10000.)
    core_rad_kpc.append(float(random.randint(50,200))/10.)
    ra.append(float(random.randint(0,500))/100000.)
    dec.append(-float(random.randint(0,500))/100000.)
    a_maj.append(float(random.randint(0,300))/100.)
    b_maj.append(float(random.randint(0,300))/100.)
    theta.append(float(random.randint(0,1)))
    
    redshift.append( zlens[i]+ float(random.randint(0,200))/100.) #lensed source must have more redshift than the lensing source
    while(redshift[i]>8):
        print("trying to get source redshift under 8")
        redshift[i] = zlens[i]+ float(random.randint(0,200))/100.
    
    magnitude.append(float(random.randint(120,220))/10.)
    
    print("===== parameter randomizing done ======")

    
    
for lens in range(hm_lenses):  
    #Actually generate the parameter files for lenstool based on the randomized values
    
    image_params(lens)
    source_params(lens)
    print("lens "+str(lens))
    
    

