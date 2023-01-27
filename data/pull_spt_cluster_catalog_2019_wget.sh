#!/bin/bash

### ~~~~~~~~~~~~ Checking if files exist first, since they are 0.5 GB total (and then unpacked) ~~~~~~~~~~~~
FILE=2500d_cluster_sample_Bocquet19.fits
if [ ! -f $FILE ]; then
	### ~~~~~~~~~~~~ Getting SPT-SZ Cluster Catalog ~~~~~~~~~~~~
	echo "Downloading SPT-SZ 2019 Cluster Catalog"
	wget -O 2500d_cluster_sample_Bocquet19.fits https://lambda.gsfc.nasa.gov/data/suborbital/SPT/cluster_2019/2500d_cluster_sample_Bocquet19.fits

	### ~~~~~~~~~~~~ Yay! No unpacking needed ~~~~~~~~~~~~
fi;
echo "SPT-SZ 2019 Cluster Catalog ready to use"
