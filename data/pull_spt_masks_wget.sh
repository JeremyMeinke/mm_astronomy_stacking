#!/bin/bash

### ~~~~~~~~~~~~ The SPT-SZ pt source mask and boundary mask

### ~~~~~~~~~~~~ Checking if files exist first ~~~~~~~~~~~~
MASK=mask_nside8192_ptsrc_50mJy.fits
BOUNDARY=mask_nside8192_bdry.fits

if [ ! -f $MASK ] && [ ! -f $BOUNDARY ]; then
	### ~~~~~~~~~~~~ Getting SPT-SZ Combined Maps ~~~~~~~~~~~~
	echo "Downloading SPT-SZ masks from Lambda.gsfc.nasa.gov, this may take some time..."
	wget -O mask_nside8192_ptsrc_50mJy.fits.gz https://lambda.gsfc.nasa.gov/data/suborbital/SPT/chown_2018/mask/mask_nside8192_ptsrc_50mJy.fits.gz
	wget -O mask_nside8192_bdry.fits.gz https://lambda.gsfc.nasa.gov/data/suborbital/SPT/chown_2018/mask_nside8192_bdry.fits.gz
	
	### ~~~~~~~~~~~~ Unpacking files ~~~~~~~~~~~~
	echo "Now extracting masks from .gz "
	gzip -d mask_nside8192_ptsrc_50mJy.fits.gz mask_nside8192_bdry.fits.gz
	
fi;
echo "SPT-SZ masks are now ready to use"
