need lenstool binaries 
change path to lenstool in "simulate_lenses" file

"with_zphot" is a table containing parameters of objects to download images of. (comments at top have been deleted)

lenstool needs  "temp_parameters.parorig" (contains parameter flags) in order to properly set params 

"dwnld_prep_makeparams.py" will download, center, size, cutouts, and generate simulated lens parameters

"merge.py" will normalize lenses to the real images, overlay, then scale pixel intensities from 0-1

the PATH variables in "merge.py" and "dwnld_prep_makeparams.py" may need to be set.



./simulate_lenses to run

