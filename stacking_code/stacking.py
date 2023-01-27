#stacking.py
###For stacking of maps in healpix (i.e. SPT or Planck) or Plate-Carree (via pixell, i.e. ACT)
from __future__ import division
import numpy as np 
import pandas as pd
import scipy.ndimage as spi
import time
import healpy as hp
from astropy.io import fits
from pixell import enmap
import gc

def gnomonic(center_dec, center_ra, pix_res, side_size = 60):		###[Deg,Deg,Arcmin,Arcmin]
	"""Creates a Gnomonic projection grid around center_dec and center_ra.  Returns grids of theta and phi.  
	NOTE: This should also already correct for instances where THETA goes outside of [0,np.pi] (which would throw an error in healpy)

	Parameters
	----------
	center_dec: int or float
		Center declination of desired grid.  Units of Degrees	(standard catalog output)
	center_ra: int or float
		Center right ascension of desired grid.  Units of Degrees	(standard catalog output)
	pix_res: int or float
		Pixel resolution of desired grid.  Units of arcminutes (as this is the scale we often care about)
	side_size: int or float
		Side size of desired grid.  Units of arcminutes (Default == 60 arcminutes)
	
	Returns
	-------
	t_set: 2d numpy.ndarray
		2d grid of theta values corresponding to gnomonic grid around center dec and ra. Resolution in units of arcminutes.
	p_set: 2d numpy.ndarray
		2d grid of theta values corresponding to gnomonic grid around center dec and ra. Resolution in units of arcminutes.
	"""
	theta = (90-center_dec) * np.pi / 180			###[Rad]
	phi = center_ra * np.pi / 180					###[Rad]

	rad = side_size / 2
	base = np.arange(-rad, rad+pix_res, pix_res) * np.pi / 180 / 60		###convert to radians here to make it stop doing annoying rounding
	dr = len(base)
	base[int(np.rint((dr-1) / 2))] = 0	###Ensuring no float precision issues at the center (for nice, really)
	###Gnomonic grid setup
	x = base * np.ones([dr, dr])
	y = (base * np.ones([dr, dr])).T[::-1]		###because latitude number goes the other direction....
	###Now to change to theta/phi notation
	rho = np.sqrt(x**2 + y**2)
	c = np.arctan2(rho, 1)
	lat = np.pi / 2 - theta
	t_set=np.pi / 2 - np.arcsin(np.cos(c)*np.sin(lat) + y*np.cos(c)*np.cos(lat))
	p_set = phi + np.arctan2(x*np.sin(c), rho*np.cos(lat)*np.cos(c) - y*np.sin(lat)*np.sin(c))
	
	return t_set, p_set

###From CMB_School_Part_03
def make_2d_gaussian_beam(N, pix_res, fwhm):
	"""Creates a 2d gaussian beam of size N x N.  pix_res and fwhm should be the same units.

	Parameters
	----------
	N: int
		Side/size of 2d grid desired.
	pix_res: int or float
		Pixel resolution of desired grid.  Same units as fwhm
	fwhm: int or float
		Gaussian fwhm desired.  Same units as pix_res
	
	Returns
	-------
	2d numpy.ndarray
		2d Gaussian grid.
	
	"""
	# make a 2d coordinate system
	N=int(N)	###Ensures integer form
	ones = np.ones(N)
	inds  = (np.arange(N) + 0.5 - N/2.) * pix_res
	X = np.outer(ones, inds)
	Y = np.transpose(X)
	R = np.sqrt(X**2. + Y**2.)
	# make a 2d gaussian 
	beam_sigma = fwhm / np.sqrt(8.*np.log(2))
	gaussian = np.exp(-.5 * (R/beam_sigma)**2.)
	gaussian = gaussian / np.sum(gaussian)
	# return the gaussian
	return gaussian

def GFilter(stamp, pix_res, high_fwhm, low_fwhm):
	"""2d gaussian filtering of a stamp cutout.

	Parameters
	----------
	stamp: numpy.ndarray
		Cutout or stamp.  Preferably of a symmetric N x N size.
	pix_res: int or float
		angular resolution of each stamp pixel.  Must have same units as map_beam_fwhm and new_beam_fwhm.
	
	Returns
	-------
	numpy.ndarray
		Gaussian filtered stamp same size as input
	"""
	f2s = 1 / (2*np.sqrt(np.log(4)))	
	high_sig = f2s * high_fwhm / pix_res
	if high_fwhm != 0:
		remove = spi.gaussian_filter(stamp, sigma=high_sig)
		new_stamp = stamp - remove
	else:
		new_stamp = np.array(stamp)
		
	low_sig = f2s * low_fwhm / pix_res
	if low_fwhm != 0:
		new_stamp = spi.gaussian_filter(new_stamp, sigma=low_sig)
	
	return new_stamp

def change_stamp_gaussian_beam(stamp, pix_res, map_beam_fwhm, new_beam_fwhm, fft_cut_both = True, cut_val = 0.25):
	"""Changes a cutout/stamp's Gaussian beam pattern.  **Doesn't take projections into account**  
	If an entire map is needed to be converted, I suggest using healpix or pixell functions.
	
	Parameters
	----------
	stamp : numpy array
		cutout or stamp. Needs to be symmetric N x N size
	pix_res: int or float
		angular resolution of each stamp pixel.  Must have same units as map_beam_fwhm and new_beam_fwhm.
	map_beam_fwhm: int or float
		Original resolution of map cut from.  Must have same units as pix_size and new_beam_fwhm.
	new_beam_fwhm: int or float
		New desired resolution of stamp.  Must have same units as pix_size and map_beam_fwhm.
	fft_cut_both: bool, optional
		Option if apply an cut in fft space to prevent spurious noise (Default == True)
	cut_val: float, optional
		Fft value to apply cut below of. (Default == 0.25)

	Returns
	-------
	numpy.ndarray
		Same size as input stamp (N x N)	
	"""
	
	###NEW NEW METHOD using fft
	if map_beam_fwhm > new_beam_fwhm:
		# make a 2d gaussian of the map beam
		map_gaussian = make_2d_gaussian_beam(len(stamp),pix_res,map_beam_fwhm)	###Since needs to match size
		new_gaussian = make_2d_gaussian_beam(len(stamp),pix_res,new_beam_fwhm)	###Since needs to match size
		# do the convolution
		map_ft_gaussian=np.fft.fft2(np.fft.fftshift(map_gaussian)) # first add the shift so that it isredeconvolved-rstacks[f] central
		new_ft_gaussian=np.fft.fft2(np.fft.fftshift(new_gaussian)) # first add the shift so that it is central
		if fft_cut_both:			###Can't decide if cutting both off at the same frequency location or at 1/sigma location is better
			new_ft_gaussian[map_ft_gaussian < cut_val] = np.min(np.real(new_ft_gaussian[map_ft_gaussian > cut_val]))
		else:
			new_ft_gaussian[map_ft_gaussian < cut_val] = cut_val
		
		map_ft_gaussian[map_ft_gaussian < cut_val] = cut_val
		ft_Map = np.fft.fft2(np.fft.fftshift(stamp)) #shift the map too
		fft_final = ft_Map * np.real(new_ft_gaussian) / np.real(map_ft_gaussian)
		deconvolved_map = np.fft.fftshift(np.real(np.fft.ifft2(fft_final)))
	else:
		deconvolved_map = GFilter(stamp, pix_res, 0, np.sqrt(new_beam_fwhm**2 - map_beam_fwhm**2))
	
	return deconvolved_map

def map_data_healpix_fits(
		file_name, hdu = 1, nside = 8192, healpix_alm = False, input_nest = False, 
		rotation = False, rot_coord = ["G", "C"], rot_lmax = None, unreadable_header = False):
	"""Load map_data from healpix-formatted fits file into a ring healpix map format (standard healpy format).
		***Careful with Planck maps, they can be in nested and galactic formats (though not always...)***
		
	Parameters
	----------

	file_name: str
		file name to load
	hdu: int, optional
		(Default == 1) If file contains multiple maps (such as T, Q, U), specify which hdu to open. Data normally starts at 1, not 0 due to first being the fits PrimaryHDU
	nside: int, optional
		healpix nside resolution of map.  Default == 8192 (resolution of SPT maps)
	healpix_alm: bool, optional
		For when provided file is a healpix alm (spherical harmonic coeffs). Default == False
	input_nest: bool, optional
		For when provided file is healpix nested format (instead of default ring). Doesn't apply to alms.  Default == False
	rotation: bool, optional
		Whether the input map is wanted to be rotated.  Default == False
	rot_coord: list, optional 
		Healpy coordinates to rotate only if rotation == True.  Default == ["G", "C"] (i.e. galactic to celestial)
	rot_lmax: int, optional
		If a lmax is to be specified for any map rotations. Default == None
	unreadable_header: bool, optional
		Set True if healpy cannot read header.  Only for maps, not alms. Default == False
	
	Returns
	-------
	1d np.ndarray
		Corresponding to healpy map formatting
	"""
	if healpix_alm:
		alm = hp.read_alm(file_name)
		if rotation:
			rot = hp.rotator.Rotator(coord = rot_coord)
			alm = rot.rotate_alm(alm, rot_lmax)
		map_data = hp.alm2map(alm, nside)
		alm = None; del alm			###attempted memory cleaning
	else:
		if unreadable_header:
			m = fits.open(file_name)
			map_data = np.asarray(np.ndarray.tolist(m[hdu].data))
			map_data = map_data.reshape(np.prod(map_data.shape),)	###As some in IDL format with two-dimensional even tho healpy does better with one dim.
			m = None; del m			###same attempt at memory cleaning
		else:
			map_data = hp.read_map(file_name, field = hdu - 1,  dtype = np.float64, nest = input_nest, memmap=True)
		###Now correcting for nest to rings
		if input_nest:
			hp.reorder(map_data, n2r = True)
	
		###Now applying rotation to map
		if rotation:
			rot = hp.rotator.Rotator(coord = rot_coord)
			map_data = rot.rotate_map_alms(map_data, rot_lmax)

	gc.collect()
	return map_data

def map_data_pixell_fits(file_name, **kwargs):
	"""Simply pixell.enmap.read_map(file_name, **kwargs)
	See https://github.com/simonsobs/pixell/blob/master/pixell/enmap.py for kwarg options"""
	return enmap.read_map(file_name, **kwargs)

def stack(
	map_data, catalog_ra, catalog_dec, side_arcmin, pix_res, data_save_name,
	plate_carree = False, nside = 8192, interpolate = True, verbose = True):
	"""Gnomonic projection stacking (sum, not average) of a catalog of points from a given map_data.
	
	Parameters
	----------

	mapdata: healpy or pixell map
		If healpy, basically just a 12 * nside**2 length 1d array.  Similar array structure for pixell plate-carree maps I think?
	catalog_ra: list or 1d array
		Right ascensions (in Degrees) of catalog to stack. Should have same length as catalog_dec
	catalog_dec: list or 1d array
		Declinations (in Degrees) of catalog to stack. Should have same length as catalog_ra
	side_arcmin: int or float
		Side length of cutout (in arcminutes).
	pix_res: int or float
		Angular resolution of each stamp pixel, in units of arcminutes
	data_save_name: str
		Save name (and location) of output data. "_%.2farcmin_%.2fpixres.txt" is added to the end
	plate_carree: bool
		False (Default) == normally expect Healpix map projection. True == If provided mapdata is in Plate-Carree form
	nside: int
		Healpix nside expected for given map (Default == 8192, for SPT maps. 12 * nside**2 = npix). Only used if plate_carree = False
	interpolate: bool
		Whether to employ bi-linear interpolation.  Takes ~4x longer, but can *slightly* reduce noise or artificial pixelization (Default == True)
	verbose: bool
		prints additional outputs for logging and checks
	
	Returns
	-------
	None (data saved as text file according to given data_save_name)

	"""
	
	if verbose:
		print('StartTime: ', time.asctime())	###For logging purposes
	
	if len(catalog_dec) != len(catalog_ra):  ###Check catalog length
		raise ValueError("lengths of catalog_dec and catalog_ra do not match!!")
	
	cat_len = len(catalog_dec)
	if verbose:
		print("Catalog length: %i"%cat_len)

	stackset = np.zeros([int(np.round(side_arcmin/pix_res) + 1), int(np.round(side_arcmin/pix_res) + 1)])		###Blank stack to start with

	if interpolate:	###bilinear interpolation
		if plate_carree:	###i.e. ACT, so using pixell defined map instead
			for c in range(cat_len):	###Cycling through catalog
				t_set, p_set = gnomonic(catalog_dec[c], catalog_ra[c], pix_res, side_size = side_arcmin)
				stackset+=map_data.at(np.asarray([np.pi/2 - t_set, p_set]), order=1)	###np.pi/2 bc mapdata.at requires declination input (not theta). bilinear interp for order=1
		else:
			for c in range(cat_len):	###Cycling through catalog
				t_set, p_set = gnomonic(catalog_dec[c], catalog_ra[c], pix_res, side_size = side_arcmin)
				stackset+=hp.get_interp_val(map_data, t_set, p_set + np.pi / 2)
	else:
		if plate_carree:
			for c in range(cat_len):	###Cycling through catalog
				t_set, p_set = gnomonic(catalog_dec[c], catalog_ra[c], pix_res, side_size = side_arcmin) 
				stackset+=map_data.at(np.asarray([np.pi/2 - t_set, p_set]), order=0)	###np.pi/2 bc mapdata.at requires declination input (not theta)
		else:
			for c in range(cat_len):	###Cycling through catalog
				t_set, p_set = gnomonic(catalog_dec[c], catalog_ra[c], pix_res, side_size = side_arcmin)
				stackset+=map_data[hp.ang2pix(nside, t_set, p_set)]
	
	savefile=data_save_name + "_%.2farcmin_%.2fpixres.txt"%(side_arcmin, pix_res)
	np.savetxt(savefile, stackset, header = "timestamp = %s"%str(time.asctime()))
	
	if verbose:
		print('savename: ', savefile)
		print('PostStack check ', time.asctime())
	
	return None

def stack_cap(
		map_data, catalog_ra, catalog_dec, cap_radii_arcmin, pix_res, data_save_name,
        plate_carree = False, nside = 8192, interpolate = True, verbose = True):
	"""Calculates the CAPs (circular apertures, as in Schaan et al. 2021) around a catalog of points from a 
	given map_data.  Uses gnomonic projection cutouts.  Pandas dataframe saved as .csv
	
	Parameters
	----------

	mapdata: healpy or pixell map
		If healpy, basically just a 12 * nside**2 length 1d array.  Similar array structure for pixell plate-carree maps I think?
	catalog_ra: list or 1d array
		Right ascensions (in Degrees) of catalog to stack. Should have same length as catalog_dec
	catalog_dec: list or 1d array
		Declinations (in Degrees) of catalog to stack. Should have same length as catalog_ra
	cap_radii_arcmin: list or numpy.ndarray
		1d list or array of CAP radii (in arcmin).
	pix_res: int or float
		Angular resolution of each stamp pixel, in units of arcminutes
	data_save_name: str
		Save name (and location) of output data.   "_CAPs_%.2fpixres.csv" is added to the end
	file_type: str
		File type to write pandas table into, i.e. DataFrame.to_"file_type" (Default == "parquet", also used if unknown file_type is given)
	plate_carree: bool
		False (Default) == normally expect Healpix map projection. True == If provided mapdata is in Plate-Carree form
	nside: int
		Healpix nside expected for given map (Default == 8192, for SPT maps. 12 * nside**2 = npix). Only used if plate_carree = False
	interpolate: bool
		Whether to employ bi-linear interpolation.  Takes ~4x longer, but can *slightly* reduce noise or artificial pixelization (Default == True)
	verbose: bool
		prints additional outputs for logging and checks
	
	Returns
	-------
	None (data saved as csv file according to given data_save_name + "_CAPs_%.2fpixres.csv"%pix_res)

	"""
	
	if verbose:
		print('StartTime: ', time.asctime())	###For logging purposes
	
	if len(catalog_dec) != len(catalog_ra):  ###Check catalog length
		raise ValueError("lengths of catalog_dec and catalog_ra do not match!!")
	
	cat_len = len(catalog_dec)
	if verbose:
		print("Catalog length: %i"%cat_len)
	
	side_arcmin = 3 * np.max(cap_radii_arcmin)	###as real max required would be 2*np.sqrt(2), i.e. < 3 

	if verbose:
		print('side arc check: ',side_arcmin)
	dr = int(np.rint(side_arcmin/pix_res + 1))
	
	def cap_ap(cap_r):
		cap_kern = np.zeros([dr, dr])
		X, Y = np.meshgrid(np.arange(dr), np.arange(dr))
		dist = np.sqrt((X - int(np.rint((dr-1) / 2)))**2 + (Y - int(np.rint((dr-1) / 2)))**2) * pix_res
		cap_kern[dist <= np.sqrt(2)*cap_r] = -pix_res**2	###negative portion of outer annulus to subtract
		cap_kern[dist <= cap_r] = pix_res**2				###positive portion of inner circle to sum
		return cap_kern

	apertures=[cap_ap(r) for r in cap_radii_arcmin]
	
	if verbose:
		print("cutout size and number of apertures: ",apertures[0].shape,len(apertures))

	ap_vals=[]
	if interpolate:	###bilinear interpolation
		if plate_carree:	###i.e. ACT, so using pixell defined map instead
			for c in range(cat_len):	###Cycling through catalog
				t_set, p_set = gnomonic(catalog_dec[c], catalog_ra[c], pix_res, side_size = side_arcmin)
				cutout=map_data.at(np.asarray([np.pi/2 - t_set, p_set]), order = 1)	 ###np.pi/2 bc mapdata.at requires declination input (not theta). bilinear interp for order=1
				temp=[]
				for ap in apertures:	###creating a list of all apertures meaured, per cutout (catalog location)
					temp.append(np.sum(cutout * ap))
				ap_vals.append(temp)
				
		else:
			for c in range(cat_len):	###Cycling through catalog
				t_set, p_set = gnomonic(catalog_dec[c], catalog_ra[c], pix_res, side_size = side_arcmin)
				cutout=hp.get_interp_val(map_data, t_set, p_set + np.pi / 2)
				temp=[]
				for ap in apertures:	###creating a list of all apertures meaured, per cutout (catalog location)
					temp.append(np.sum(cutout * ap))
				ap_vals.append(temp)
	else:
		if plate_carree:
			for c in range(cat_len):	###Cycling through catalog
				t_set, p_set = gnomonic(catalog_dec[c], catalog_ra[c], pix_res, side_size = side_arcmin) 
				cutout=map_data.at(np.asarray([np.pi/2 - t_set, p_set]), order = 0)	###np.pi/2 bc mapdata.at requires declination input (not theta)
				temp=[]
				for ap in apertures:	###creating a list of all apertures meaured, per cutout (catalog location)
					temp.append(np.sum(cutout * ap))
				ap_vals.append(temp)
		else:
			for c in range(cat_len):	###Cycling through catalog
				t_set, p_set = gnomonic(catalog_dec[c], catalog_ra[c], pix_res, side_size = side_arcmin)
				cutout=map_data[hp.ang2pix(nside, t_set, p_set)]
				temp=[]
				for ap in apertures:	###creating a list of all apertures meaured, per cutout (catalog location)
					temp.append(np.sum(cutout * ap))
				ap_vals.append(temp)
	
	###into pandas table
	ap_vals = pd.DataFrame(ap_vals, columns = ["%.3f"%cap + " [Arcmin]" for cap in cap_radii_arcmin])
	###And save to a csv table.  
	savefile = data_save_name + "_CAPs_%.2fpixres.csv"%(pix_res)
	ap_vals.to_csv(savefile, sep=",", na_rep="  ")
	# Personally I've preferred pickle (.pkl) for my work due to better read/write speed, but it has sharing/backward compatability issues
	
	if verbose:
		print('savename: ', savefile)
		print('PostStack check ', time.asctime())
	
	return None

def stack_radial_avg(
		map_data, catalog_ra, catalog_dec, rad_avg_radii_arcmin, pix_res, data_save_name,
    	plate_carree = False, nside = 8192, padding = 2, subtract_mean = True, interpolate = True, verbose = True):
	"""Calculates the radial averages between provided radii around a catalog of points from a given map_data.  
	Uses gnomonic projection cutouts.  Pandas dataframe saved as .csv
	
	Parameters
	----------

	mapdata: healpy or pixell map
		If healpy, basically just a 12 * nside**2 length 1d array.  Similar array structure for pixell plate-carree maps I think?
	catalog_ra: list or 1d array
		Right ascensions (in Degrees) of catalog to stack. Should have same length as catalog_dec
	catalog_dec: list or 1d array
		Declinations (in Degrees) of catalog to stack. Should have same length as catalog_ra
	rad_avg_radii_arcmin: list or numpy.ndarray
		1d list or array of radial average radii edges/bounds (in arcmin).  Should have length >= 2
	pix_res: int or float
		Angular resolution of each stamp pixel, in units of arcminutes
	data_save_name: str
		Save name (and location) of output data.   "_radial_avg_%.2fpixres.csv" is added to the end
	file_type: str
		File type to write pandas table into, i.e. DataFrame.to_"file_type" (Default == "parquet", also used if unknown file_type is given)
	plate_carree: bool
		False (Default) == normally expect Healpix map projection. True == If provided mapdata is in Plate-Carree form
	nside: int
		Healpix nside expected for given map (Default == 8192, for SPT maps. 12 * nside**2 = npix). Only used if plate_carree = False
	padding: int or float
		(Default == 2) Extra padding in arcmin to add to cutout size. side_arcmin = 2*np.max(rad_avg_radii_arcmin) + padding
	subtract_mean: bool
		Subtracts mean of each cutout before applying apertures.  This can reduce large-scale flucuations (i.e. foregrounds, Default == True)
	interpolate: bool
		Whether to employ bi-linear interpolation.  Takes ~4x longer, but can *slightly* reduce noise or artificial pixelization (Default == True)
	verbose: bool
		prints additional outputs for logging and checks
	
	Returns
	-------
	None (data saved as csv file according to given data_save_name + "_CAPs_%.2fpixres.csv"%pix_res)

	"""
	
	if verbose:
		print('StartTime: ', time.asctime())	###For logging purposes
	
	###Dimension Checks
	if len(catalog_dec) != len(catalog_ra):  ###Check catalog length
		raise ValueError("lengths of catalog_dec and catalog_ra do not match!!")
	if len(rad_avg_radii_arcmin) < 2:
		raise ValueError("rad_avg_radii_arcmin must be at least length = 2 (i.e. defined EDGES to average between)")
	
	cat_len = len(catalog_dec)
	if verbose:
		print("Catalog length: %i"%cat_len)
	
	side_arcmin = 2*np.max(rad_avg_radii_arcmin) + padding		###as real max required would be 2*np.max(), so optional for extra padding

	if verbose:
		print('side arc check: ',side_arcmin)
	dr = int(np.rint(side_arcmin/pix_res + 1))

	def radial_avg_ap(r_min, r_max):
		rad_avg_kern = np.full([dr, dr], np.nan)
		X, Y = np.meshgrid(np.arange(dr), np.arange(dr))
		dist = np.sqrt((X - int(np.rint((dr-1) / 2)))**2 + (Y - int(np.rint((dr-1) / 2)))**2) * pix_res
		rad_avg_kern[(dist >= r_min) & (dist < r_max)] = 1		###Changing values equal/above rmin (to include 0), and below rmax
		return rad_avg_kern

	apertures=[radial_avg_ap(rad_avg_radii_arcmin[r], rad_avg_radii_arcmin[r + 1]) for r in range(len(rad_avg_radii_arcmin) - 1)]	###As radial edges are given
	if verbose:
		print("cutout size and number of apertures: ",apertures[0].shape, len(apertures))

	columns = ["%.2f to %.2f [Arcmin]"%(rad_avg_radii_arcmin[r], rad_avg_radii_arcmin[r + 1]) for r in range(len(rad_avg_radii_arcmin) - 1)]
	ap_vals=[]
	###I despise this nesting of 3 if statements, but it seems like the method that doesn't introduce any unnecessary slowing that'd occur inside the for loops
	# Feel free to modify this if there's a better way for you
	if subtract_mean:
		if interpolate:	###bilinear interpolation
			if plate_carree:	###i.e. ACT, so using pixell defined map instead
				for c in range(cat_len):	###Cycling through catalog
					t_set, p_set = gnomonic(catalog_dec[c], catalog_ra[c], pix_res, side_size = side_arcmin)
					cutout = map_data.at(np.asarray([np.pi/2 - t_set, p_set]), order = 1)	 ###np.pi/2 bc mapdata.at requires declination input (not theta). bilinear interp for order=1
					cutout -= np.mean(cutout)
					temp=[]
					for ap in apertures:	###creating a list of all apertures meaured, per cutout (catalog location)
						temp.append(np.nanmean(cutout * ap))
					ap_vals.append(temp)
					
			else:
				for c in range(cat_len):	###Cycling through catalog
					t_set, p_set = gnomonic(catalog_dec[c], catalog_ra[c], pix_res, side_size = side_arcmin)
					cutout = hp.get_interp_val(map_data, t_set, p_set + np.pi / 2)
					cutout -= np.mean(cutout)
					temp=[]
					for ap in apertures:	###creating a list of all apertures meaured, per cutout (catalog location)
						temp.append(np.nanmean(cutout * ap))
					ap_vals.append(temp)
		else:
			if plate_carree:
				for c in range(cat_len):	###Cycling through catalog
					t_set, p_set = gnomonic(catalog_dec[c], catalog_ra[c], pix_res, side_size = side_arcmin) 
					cutout = map_data.at(np.asarray([np.pi/2 - t_set, p_set]), order = 0)	###np.pi/2 bc mapdata.at requires declination input (not theta)
					cutout -= np.mean(cutout)
					temp=[]
					for ap in apertures:	###creating a list of all apertures meaured, per cutout (catalog location)
						temp.append(np.nanmean(cutout * ap))
					ap_vals.append(temp)
			else:
				for c in range(cat_len):	###Cycling through catalog
					t_set, p_set = gnomonic(catalog_dec[c], catalog_ra[c], pix_res, side_size = side_arcmin)
					cutout = map_data[hp.ang2pix(nside, t_set, p_set)]
					cutout -= np.mean(cutout)
					temp=[]
					for ap in apertures:	###creating a list of all apertures meaured, per cutout (catalog location)
						temp.append(np.nanmean(cutout * ap))
					ap_vals.append(temp)
	else:	###no need to mean subtract, this should make it slighty? faster
		if interpolate:	###bilinear interpolation
			if plate_carree:	###i.e. ACT, so using pixell defined map instead
				for c in range(cat_len):	###Cycling through catalog
					t_set, p_set = gnomonic(catalog_dec[c], catalog_ra[c], pix_res, side_size = side_arcmin)
					cutout = map_data.at(np.asarray([np.pi/2 - t_set, p_set]), order = 1)	 ###np.pi/2 bc mapdata.at requires declination input (not theta). bilinear interp for order=1
					temp=[]
					for ap in apertures:	###creating a list of all apertures meaured, per cutout (catalog location)
						temp.append(np.nanmean(cutout * ap))
					ap_vals.append(temp)
					
			else:
				for c in range(cat_len):	###Cycling through catalog
					t_set, p_set = gnomonic(catalog_dec[c], catalog_ra[c], pix_res, side_size = side_arcmin)
					cutout = hp.get_interp_val(map_data, t_set, p_set + np.pi / 2)
					temp=[]
					for ap in apertures:	###creating a list of all apertures meaured, per cutout (catalog location)
						temp.append(np.nanmean(cutout * ap))
					ap_vals.append(temp)
		else:
			if plate_carree:
				for c in range(cat_len):	###Cycling through catalog
					t_set, p_set = gnomonic(catalog_dec[c], catalog_ra[c], pix_res, side_size = side_arcmin) 
					cutout = map_data.at(np.asarray([np.pi/2 - t_set, p_set]), order = 0)	###np.pi/2 bc mapdata.at requires declination input (not theta)
					temp=[]
					for ap in apertures:	###creating a list of all apertures meaured, per cutout (catalog location)
						temp.append(np.nanmean(cutout * ap))
					ap_vals.append(temp)
			else:
				for c in range(cat_len):	###Cycling through catalog
					t_set, p_set = gnomonic(catalog_dec[c], catalog_ra[c], pix_res, side_size = side_arcmin)
					cutout = map_data[hp.ang2pix(nside, t_set, p_set)]
					temp=[]
					for ap in apertures:	###creating a list of all apertures meaured, per cutout (catalog location)
						temp.append(np.nanmean(cutout * ap))
					ap_vals.append(temp)


	###into pandas table
	ap_vals = pd.DataFrame(ap_vals, columns = columns)
	###And save to a csv table.  
	savefile = data_save_name + "_radial_avg_%.2fpixres.csv"%(pix_res)
	ap_vals.to_csv(savefile, sep=",", na_rep="  ")
	# Personally I've preferred pickle (.pkl) for my work due to better read/write speed, but it has sharing/backward compatability issues

	if verbose:
		print('savename: ',savefile)
		print('PostStack check ',time.asctime())
	
	return None