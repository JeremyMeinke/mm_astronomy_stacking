#!/bin/bash

### ~~~~~~~~~~~~ Grabbing the pixel weights as recommended by the healpy documentation for the most accurate alm <--> map transformations

### ~~~~~~~~~~~~ Checking if files exist first ~~~~~~~~~~~~
PIX2048=healpix_full_weights_nside_2048.fits
PIX8192=healpix_full_weights_nside_8192.fits

if [ ! -f $PIX2048 ]; then
	### ~~~~~~~~~~~~ Getting Healpy Pixel Weights ~~~~~~~~~~~~
	echo "Downloading healpy 2048 pixel weights..."
	wget -O healpix_full_weights_nside_2048.fits https://github.com/healpy/healpy-data/raw/master/full_weights/healpix_full_weights_nside_2048.fits
	
fi;

if [ ! -f $PIX8192 ]; then
	### ~~~~~~~~~~~~ Getting Healpy Pixel Weights ~~~~~~~~~~~~
	echo "Downloading healpy 8192 pixel weights..."
	wget -O healpix_full_weights_nside_8192.fits https://github.com/healpy/healpy-data/releases/download/full_weights/healpix_full_weights_nside_8192.fits
	
fi;

echo "Healpy pixel weights now downloaded"
