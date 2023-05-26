#!/bin/bash

### ~~~~~~~~~~~~ The SMICA Planck CMB no-SZ map

### ~~~~~~~~~~~~ Checking if files exist first ~~~~~~~~~~~~
MAP=COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits


if [ ! -f $MAP ]; then
	### ~~~~~~~~~~~~ Getting SPT-SZ Combined Maps ~~~~~~~~~~~~
	echo "Downloading Planck (SMICA) CMB no-SZ map from Lambda.gsfc.nasa.gov..."
	wget -O COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits

	
fi;
echo "Planck (SMICA) CMB no-SZ map is now ready to use"
