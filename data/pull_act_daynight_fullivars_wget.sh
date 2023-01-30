#!/bin/bash

### ~~~~~~~~~~~~ Checking if files exist first, since these are 1 GB total (they unfortunately didn't pack them any) ~~~~~~~~~~~~
FILE90=act_planck_dr5.01_s08s18_AA_f090_daynight_fullivar.fits
FILE150=act_planck_dr5.01_s08s18_AA_f150_daynight_fullivar.fits
FILE220=act_planck_dr5.01_s08s18_AA_f220_daynight_fullivar.fits
if [ ! -f $FILE90 ] && [ ! -f $FILE150 ] && [ ! -f $FILE220 ]; then
	### ~~~~~~~~~~~~ Getting ACT + Planck Day-Night Full ivar (inverse variance) maps ~~~~~~~~~~~~
	echo "Downloading ACT+Planck DayNight full ivar maps from Lambda.gsfc.nasa.gov, this may take some time..."
	wget -O act_planck_dr5.01_s08s18_AA_f090_daynight_fullivar.fits https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr5/maps/act_planck_dr5.01_s08s18_AA_f090_daynight_fullivar.fits
	wget -O act_planck_dr5.01_s08s18_AA_f150_daynight_fullivar.fits https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr5/maps/act_planck_dr5.01_s08s18_AA_f150_daynight_fullivar.fits
	wget -O act_planck_dr5.01_s08s18_AA_f220_daynight_fullivar.fits https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr5/maps/act_planck_dr5.01_s08s18_AA_f220_daynight_fullivar.fits
	
	### ~~~~~~~~~~~~ No packing needed... yay? ~~~~~~~~~~~~
	
fi;
echo "ACT+Planck DayNight full ivar maps now ready to use"
