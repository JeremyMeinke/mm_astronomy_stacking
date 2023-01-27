#!/bin/bash

### ~~~~~~~~~~~~ I often find the SPT-SZ alm data to be less memory-intensive than the map format (not entirely sure why, something with healpy)

### ~~~~~~~~~~~~ Checking if files exist first, since they are 0.5 GB total (and then unpacked) ~~~~~~~~~~~~
FILE95=alm_combined_data_95GHz_ptsrcmasked_50mJy.fits
FILE150=alm_combined_data_150GHz_ptsrcmasked_50mJy.fits
FILE220=alm_combined_data_220GHz_ptsrcmasked_50mJy.fits
if [ ! -f $FILE95 ] && [ ! -f $FILE150 ] && [ ! -f $FILE220 ]; then
	### ~~~~~~~~~~~~ Getting SPT-SZ Combined Maps ~~~~~~~~~~~~
	echo "Downloading SPT-SZ Combined alms (spherical harmonics) from Lambda.gsfc.nasa.gov, this may take some time..."
	wget -O alm_combined_data_95GHz_ptsrcmasked_50mJy.fits.gz https://lambda.gsfc.nasa.gov/data/suborbital/SPT/chown_2018/alms/alm_combined_data_95GHz_ptsrcmasked_50mJy.fits.gz
	wget -O alm_combined_data_150GHz_ptsrcmasked_50mJy.fits.gz https://lambda.gsfc.nasa.gov/data/suborbital/SPT/chown_2018/alms/alm_combined_data_150GHz_ptsrcmasked_50mJy.fits.gz
	wget -O alm_combined_data_220GHz_ptsrcmasked_50mJy.fits.gz https://lambda.gsfc.nasa.gov/data/suborbital/SPT/chown_2018/alms/alm_combined_data_220GHz_ptsrcmasked_50mJy.fits.gz
	
	### ~~~~~~~~~~~~ Unpacking files ~~~~~~~~~~~~
	echo "Now extracting alms from .gz "
	gzip -d alm_combined_data_95GHz_ptsrcmasked_50mJy.fits.gz alm_combined_data_150GHz_ptsrcmasked_50mJy.fits.gz alm_combined_data_220GHz_ptsrcmasked_50mJy.fits.gz
	
fi;
echo "SPT-SZ combined alms now ready to use"
