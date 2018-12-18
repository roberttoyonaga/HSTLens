#!/bin/sh
echo "Generating simulations"

echo "How many lenses do you want to build from param files?"
read hm_lenses

echo $hm_lenses | python only_make_lenses.py

COUNTER=0
while [  $COUNTER -lt $hm_lenses ]; do
    echo Built $COUNTER
    /home/toyonagar/anaconda3/HST_Lens/lenstool-7.1-linux64/lenstool ./temp_parameters_$COUNTER.par -n
    let COUNTER=COUNTER+1 
done

rm *.par *.cat



