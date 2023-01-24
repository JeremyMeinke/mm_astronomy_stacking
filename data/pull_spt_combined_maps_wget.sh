#!/bin/bash

### ~~~~~~~~~~~~ Checking if files exist first, since they are 0.5 GB total (and then unpacked) ~~~~~~~~~~~~
FILE95=combined_map_95GHz_nside8192_ptsrcmasked_50mJy.fits
FILE150=combined_map_150GHz_nside8192_ptsrcmasked_50mJy.fits
FILE220=combined_map_220GHz_nside8192_ptsrcmasked_50mJy.fits
if [ ! -f $FILE95 ] && [ ! -f $FILE150 ] && [ ! -f $FILE220 ]; then
	### ~~~~~~~~~~~~ Getting SPT-SZ Combined Maps ~~~~~~~~~~~~
	echo "Downloading SPT-SZ Combined Maps from Lambda.gsfc.nasa.gov, this may take some time..."
	wget -O combined_map_95GHz_nside8192_ptsrcmasked_50mJy.fits.gz https://lambda.gsfc.nasa.gov/data/suborbital/SPT/chown_2018/maps/combined_map_95GHz_nside8192_ptsrcmasked_50mJy.fits.gz
	wget -O combined_map_150GHz_nside8192_ptsrcmasked_50mJy.fits.gz https://lambda.gsfc.nasa.gov/data/suborbital/SPT/chown_2018/maps/combined_map_150GHz_nside8192_ptsrcmasked_50mJy.fits.gz
	wget -O combined_map_220GHz_nside8192_ptsrcmasked_50mJy.fits.gz https://lambda.gsfc.nasa.gov/data/suborbital/SPT/chown_2018/maps/combined_map_220GHz_nside8192_ptsrcmasked_50mJy.fits.gz
	
	### ~~~~~~~~~~~~ Unpacking files ~~~~~~~~~~~~
	echo "Now extracting maps from .gz "
	gzip -d combined_map_95GHz_nside8192_ptsrcmasked_50mJy.fits.gz combined_map_150GHz_nside8192_ptsrcmasked_50mJy.fits.gz combined_map_220GHz_nside8192_ptsrcmasked_50mJy.fits.gz 
	
fi;
echo "SPT-SZ combined maps now ready to use"
