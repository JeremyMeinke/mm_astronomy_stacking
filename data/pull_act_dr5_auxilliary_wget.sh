#!/bin/bash

### ~~~~~~~~~~~~ Checking if files exist first, since they are 0.5 GB total (and then unpacked) ~~~~~~~~~~~~
FOLDER=act_dr5.01_auxilliary
if [ ! -d $FOLDER ]; then
	### ~~~~~~~~~~~~ Getting ACT Auxilliary products (i.e. bandpasses, most importantly) ~~~~~~~~~~~~
	echo "Downloading ACT Auxilliary Products"
	wget act_dr5.01_auxilliary.zip https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr5/maps/act_dr5.01_auxilliary.zip

	### ~~~~~~~~~~~~ Unpacking ~~~~~~~~~~~~
	unzip act_dr5.01_auxilliary.zip -d act_dr5.01_auxilliary
	rm act_dr5.01_auxilliary.zip
fi;
echo "ACT Auxilliary Products ready to use"
