Usage Instructions
Step 1: Creating training set
 - 1) /make_simulations
    - "with_zphot" is a table containing parameters of objects to download images of. (comments at top have been deleted)                                                                                                            
    - lenstool needs  "temp_parameters.parorig" (contains parameter flags) in order to properly set params                                                                                                                           
    - "dwnld_prep_makeparams.py" will download, center, size, cutouts, and generate simulated lens parameters                                              
    - "merge_psf_convolve.py" will normalize lenses to the real images, overlay, then scale pixel intensities from 0-1                                                   
    - the PATH variables in "merge_psf_convolve.py" and "dwnld_prep_makeparams.py" may need to be set.         

    - ensure "with_zphot.tsv" is in the same directory. This is the catalog that will be used to create lens simulations knowing the redshift of the source. 
    - "temp_parameters.parorig" must also be in the directory for lenstool to work
    - need lenstool binaries. change path to lenstool in "simulate_lenses" file
    - change the path to save original image cutouts, centered image cutouts, created lensed features in "dwnl_prep_makeparams.py"
    - change the path to the created lenses, centered image cutouts, and completed simulations in merge_psf_convolve.py
    - To create simulations run ./simulate_lenses and provide the number of lenses to create when prompted.
    - To create negatives run "python get_negatives.py" and follow prompt

=====+++++======Alternatively========++++++========
   
    - use make_training_set_batchjob_3.ipynb to do the merging and creation of negatives (this method was used to create batchjob3 training set)
    - You must already have created the lens featurs and centered cutouts.

- 2) train the model /CC_batch_processing
    - /batchjob_3
    - HST_Lens_training_script_batchjob3.ipynb
    - 3 models should be trained and their weigths kept in "batchjob3_weights"

- 3) Run detections using batch processing
    - "jobscript_3.sh" launches 1472 24h jobs each with 1cpu and 8000M of memory. Each job is tasked with running detections on 5 HST images
    - "pipeline_3.exe" is called by jobscript_3.sh and is responsible for launching the scripts to prepare and run detecitons on images
    - "dwnld_3.py" downloads images based on ../test_csv and calls source extractor on them
    - "run_predictions.py" creates predictions with 3 different models and averages+outputs their ratings 
    - ""../test_csv" list of all HST co-added product images as of november 2018

    - change DWNLD_PATH variable (the full large HST image) in "dwnld_3.py" (and CSV_PATH too if needed). Also change path to .cat file in call to sextractor
    - change IMAGE_PATH(full HST image), OUT_PATH(where to send results files), SE_PATH(sextractor catalogs), WEIGHTS_PATH(model weights), CSV_PATH(list of HST images,if needed) in "run_predictions_3.py"


    - to run detections: sbatch jobscript_3.sh 
        - You can also test out one job by using "python pipeline_3.exe" then entering a starting point from 0-7000. It will start do predicitons on a set of 5 HST images starting with that specific index






