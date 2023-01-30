#!/bin/bash

### ~~~~~~~~~~~~ Checking if files exist first, since these are 5 GB total (they unfortunately didn't pack them any) ~~~~~~~~~~~~
FILE90=act_planck_dr5.01_s08s18_AA_f090_daynight_map_srcfree.fits
FILE150=act_planck_dr5.01_s08s18_AA_f150_daynight_map_srcfree.fits
FILE220=act_planck_dr5.01_s08s18_AA_f220_daynight_map_srcfree.fits
if [ ! -f $FILE90 ] && [ ! -f $FILE150 ] && [ ! -f $FILE220 ]; then
	### ~~~~~~~~~~~~ Getting ACT + Planck Day-Night SrcFree maps ~~~~~~~~~~~~
	echo "Downloading ACT+Planck DayNight SrcFree Maps from Lambda.gsfc.nasa.gov, this may take some time..."
	wget -O act_planck_dr5.01_s08s18_AA_f090_daynight_map_srcfree.fits https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr5/maps/act_planck_dr5.01_s08s18_AA_f090_daynight_map_srcfree.fits
	wget -O act_planck_dr5.01_s08s18_AA_f150_daynight_map_srcfree.fits https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr5/maps/act_planck_dr5.01_s08s18_AA_f150_daynight_map_srcfree.fits
	wget -O act_planck_dr5.01_s08s18_AA_f220_daynight_map_srcfree.fits https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr5/maps/act_planck_dr5.01_s08s18_AA_f220_daynight_map_srcfree.fits
	
	### ~~~~~~~~~~~~ No packing needed... yay? ~~~~~~~~~~~~
	
fi;
echo "ACT+Planck DayNight SrcFree maps now ready to use"
