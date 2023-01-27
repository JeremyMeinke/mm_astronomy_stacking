from __future__ import division
import numpy as np
import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.integrate as spint
import scipy.ndimage as spi
from scipy.optimize import curve_fit
from astropy.cosmology import LambdaCDM
import pandas as pd
import math
import scipy.special as spsp
import time
import warnings
import gc
from typing import Any, Callable



def angdiff(RA1, DEC1, RA2, DEC2, degrees = False):
	"""angle betwix locations.  Set degrees = True if input/output are in degrees"""
	if degrees:
		RA1 = np.deg2rad(RA1); DEC1 = np.deg2rad(DEC1); RA2 = np.deg2rad(RA2); DEC2 = np.deg2rad(DEC2)
		a = 2 * np.arcsin(np.sqrt(np.sin((DEC1-DEC2) / 2)**2 + np.cos(DEC1)*np.cos(DEC2)*np.sin((RA1-RA2) / 2)**2)) ##angle betwix locations, in radians
		return np.rad2deg(a)
	else:
		a = 2 * np.arcsin(np.sqrt(np.sin((DEC1-DEC2) / 2)**2 + np.cos(DEC1)*np.cos(DEC2)*np.sin((RA1-RA2) / 2)**2)) ##angle betwix locations, in radians
		return a

def df_compare_dup_indices(df1, df2, cols):	###Identifies identical locations of 2 catalogs that may have different indices
	'''To find indices (by both dfs) for rows that appear in both dfs, 
	ONLY VALID IF df1 AND df2 DON'T INDIVIDUALLY CONTAIN ANY DUPLICATES ALREADY'''
	###For first df
	df_comb1 = pd.concat([df2, df1]) ###add df1 to df2
	df1_inds = df_comb1[df_comb1.duplicated(cols) == True].index	###find indices for duplicates (that would exist only in the added df1 part)

	df_comb2=pd.concat([df1, df2]) ###add df2 to df1
	df2_inds=df_comb2[df_comb2.duplicated(cols) == True].index	###find indices for duplicates (that would exist only in the added df2 part)
	
	return df1_inds, df2_inds

def df_cat_within(df, ra_mins, ra_maxs, dec_mins, dec_maxs, col_start = 0):	###Grabs subset of catalog within a given RA and DEC range
	'''Select Assuming RA and DEC are the first [0] and second [1] columns of dataframe '''
	cols=list(df.columns)
	print("Columns: ", cols, "\n")
	if (len(ra_mins) != len(ra_maxs)) or (len(dec_mins) != len(dec_maxs)) or (len(ra_mins) != len(dec_mins)):
		print("Equal number of RA and DEC bounds required (Make squares)")
	else:
		bnd_bools = []
		for i in range(len(ra_mins)):
			bnd_bools.append((df[cols[col_start]]%360 >= ra_mins[i]%360) & (df[cols[col_start]]%360 <= ra_maxs[i]%360) & (df[cols[1 + col_start]] >= dec_mins[i]) & (df[cols[1 + col_start]] <= dec_maxs[i]))
		bnd_bools = np.asarray(bnd_bools)
		new_df = df[np.any(bnd_bools, axis = 0)]
	return new_df

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

def remove_gradient(image, pix_res = 0.05, center_cut_dist = 0):
	"""Uses GLS to fit and remove a gradient from 2d matrix/image.
	Where center_cut_dist == distance from center to cut (center_cut_dist/pix_res pixels)
	Returns same sized array as input image
	
	Parameters
	----------
	
	image: numpy.ndarray
		2D numpy array to have gradient removed
	pix_res: float, optional
		pixel resolution of image.  Only used if cutting a central signal/source desired not the fit in gradient
	center_cut_dist: int or float, optional
		Distance from the center desired to cut (same units as pix_res).  The cut distance will not be fit to gradient
	
	Returns
	-------
	2D numpy.ndarray, same size as image
	
	"""

	DR=len(image)
	X, Y = np.meshgrid(np.arange(DR), np.arange(DR))
	mask_kern = np.ones([DR, DR])

	###Make center cut if required
	if center_cut_dist > 0:
		dist = np.sqrt((X - int(np.rint((DR-1) / 2)))**2 + (Y - int(np.rint((DR-1) / 2)))**2) * pix_res
		mask_kern[dist < center_cut_dist] = np.nan
	
	mask_image = mask_kern * image
	###Now to average along each direction, fit a linear slope, and remove from image
	###axis=0 first (X)
	avg_x = np.nanmean(mask_image, axis = 0)
	x_matrix = np.column_stack([X[0,:], np.ones_like(avg_x)])
	y_vector = np.array([avg_x,]).T
	N = len(y_vector)
	fit = np.linalg.inv(x_matrix.T @ x_matrix) @ x_matrix.T @ y_vector
	# scale = (y_vector - x_matrix@fit).T @ (y_vector - x_matrix@fit) / (N - len(fit))
	# fitcov = np.linalg.inv(x_matrix.T @ x_matrix) * scale
	# print("X fit and cov: ",fit,fitcov)
	image = image - (X*fit[0] + fit[1])

	###axis=1 Second (Y)
	mask_image = mask_image - (X*fit[0] + fit[1])
	avg_y = np.nanmean(mask_image, axis = 1)
	x_matrix = np.column_stack([Y[:,0], np.ones_like(avg_y)])
	y_vector = np.array([avg_y,]).T
	N = len(y_vector)
	fit = np.linalg.inv(x_matrix.T @ x_matrix) @ x_matrix.T @ y_vector
	# scale = (y_vector - x_matrix@fit).T @ (y_vector - x_matrix@fit) / (N - len(fit))
	# fitcov=np.linalg.inv(x_matrix.T @ x_matrix) * scale
	# print("Y fit and cov: ",fit,fitcov)
	image = image - (Y*fit[0] + fit[1])
	return image

def radial_avg(stamp, rad_avg_radii, pix_res = 0.05, pix_scale = False):
	"""Calculate radial average(s) on a given stamp, for desired radial edges.  
	Returns a list of radial averages.
	
	Parameters
	----------
	stamp: numpy.ndarray
		2D numpy array stamp to be measured
	rad_avg_radii: list or numpy.ndarray
		List or 1d array of desired radial average edges to calculate within.  Must be same units as pix_res
	pix_res: float, optional
		Pixel resolution of stamp.  Must be same units as rad_avg_radii (Default == 0.05)
	pix_scale: bool, optional
		Pixel scaling, i.e. if desired to be scaled by pixel area (Default == False, since it's an average)
	
	Returns
	-------
	list, of size len(rad_avg_radii) - 1

	"""
	if pix_scale:
		scale = pix_res**2
	else:
		scale = 1
	
	DR=len(stamp)
	def radial_avg_ap(r_min, r_max):
		rad_avg_kern = np.full([DR, DR], np.nan)
		X, Y = np.meshgrid(np.arange(DR), np.arange(DR))
		dist = np.sqrt((X - int(np.rint((DR-1) / 2)))**2 + (Y - int(np.rint((DR-1) / 2)))**2) * pix_res
		rad_avg_kern[(dist >= r_min) & (dist < r_max)] = scale		###Changing values equal/above rmin (to include 0), and below rmax
		return rad_avg_kern
	
	rad_avgs = [np.nanmean(stamp*radial_avg_ap(rad_avg_radii[r], rad_avg_radii[r+1])) for r in range(len(rad_avg_radii) - 1)]
	return rad_avgs

def gaussian_aperture_sum(stamp, pix_res, fwhm):
	"""Sum within a (normalized) gaussian aperture on a given stamp.  pix_res and fwhm should have the same units.
	Returns float"""
	DR = len(stamp)
	gaussian_kernel = make_2d_gaussian_beam(DR, pix_res, fwhm)
	return np.sum(gaussian_kernel * stamp)

def tophat_aperture_sum(stamp, pix_res, th_rad, pix_scale = True):
	"""Returns tophat sum around given stamp. pix_res and th_rad should have the same units.
	if pix_scale == True, takes pixel area/resolution into account"""

	if pix_scale:
		scale = pix_res**2
	else:
		scale = 1
	
	DR = len(stamp)
	tophat_kern = np.full([DR, DR], np.nan)
	X, Y = np.meshgrid(np.arange(DR), np.arange(DR))
	dist = np.sqrt((X - int(np.rint((DR-1) / 2)))**2 + (Y - int(np.rint((DR-1) / 2)))**2) * pix_res
	tophat_kern[dist <= th_rad] = scale
	return np.nansum(tophat_kern * stamp)

###Tophat minus annulus aperture sum like the ACT paper (Schaan, et. al., 2021)
def cap_circular_aperture_sum(stamp, pix_res, cap_rad, pix_scale = True):
	"""Returns CAP sum around given stamp. pix_res and cap_rad should have the same units.
	if pix_scale == True, takes pixel area/resolution into account"""
	
	if pix_scale:
		scale = pix_res**2
	else: 
		scale = 1
	
	DR = len(stamp)
	cap_kern = np.full([DR, DR], np.nan)
	X, Y = np.meshgrid(np.arange(DR), np.arange(DR))
	dist = np.sqrt((X - int(np.rint((DR-1) / 2)))**2 + (Y - int(np.rint((DR-1) / 2)))**2) * pix_res
	cap_kern[dist <= np.sqrt(2) * cap_rad] = -scale
	cap_kern[dist <= cap_rad] = scale
	return np.nansum(cap_kern * stamp)

def noise(Image, pixres, innerdrop_rad, outerdrop_rad):
	l = len(Image)
	c = int((l-1)/2)
	add = 0
	mean=0
	npix = 0
	if outerdrop_rad == 0:
		for x in range(l):
			for y in range(l):
				dist = np.sqrt((x-c)**2+(y-c)**2)*pixres
				if dist >= innerdrop_rad:
					add += Image[x,y]**2
					mean+=Image[x,y]
					npix += 1

	else:
		for x in range(l):
			for y in range(l):
				dist = np.sqrt((x-c)**2+(y-c)**2)*pixres
				if dist >= innerdrop_rad and dist <= outerdrop_rad:
					add += Image[x,y]**2
					mean+=Image[x,y]
					npix += 1
	std = np.sqrt((add/npix-(mean/npix)**2)*npix/(npix-1))*10**6	##Final bit for sample variance, for most images in question won't change anything....
	#print(npix)
	#print(rms)
	return np.round(std, 4), np.round(mean/npix*10**6,4)

###Gaussian Fitting image and subtracting from center, determine noise from remainder
def GFit(Image, pix_res, fwhm, qplot, fwhm_max):	##fwhm estimate for fit (qplot True/False for if quick plot desired)
	l = len(Image)
	c = int((l-1)/2)
	RAD = c*pix_res
	pmin = np.min(Image)
	pmax = np.max(Image)
	BASE = np.arange(-RAD, RAD + pix_res, pix_res) #Base array for stack mesh creation
	#	print(BASE)
	MESH1, MESH2 = np.meshgrid(BASE, BASE)
	xdata = np.vstack((MESH1.ravel(), MESH2.ravel()) )
	f2s = 1 / (2 * np.sqrt(np.log(4)) )
	
	def Gaussian(x, y, peak,sig):
		# peak=args[0]
		# sig=args[1]
		return peak * np.exp(-((x)**2+(y)**2)/2/sig**2)

	def G(M,peak,sig):
		x, y = M
		sol = np.zeros(x.shape)
		sol += Gaussian(x,y,peak,sig)
		return sol

	##Initial guess:
	p0 = (Image[c,c], fwhm*f2s)
	bnds=([-np.inf,0.01],[np.inf,fwhm_max*f2s])
	popt, pcov = curve_fit(G, xdata, Image.ravel(), p0=p0,bounds=bnds)[:2]
	# print(popt)
	# print(pcov)
	if qplot:
		residual = Image - Gaussian(MESH1, MESH2, *popt)
		plt.figure(figsize=(10, 4))
		plt.subplot(1, 3, 1)
		plt.imshow(Image, origin='lower', interpolation='nearest', vmin = pmin, vmax = pmax, cmap = 'jet')
		plt.title("Data")
		plt.subplot(1, 3, 2)
		plt.imshow(Gaussian(MESH1, MESH2, *popt), origin='lower', interpolation='nearest', vmin = pmin, vmax = pmax, cmap = 'jet')
		plt.title("Model")
		plt.subplot(1, 3, 3)
		plt.imshow(residual, origin='lower', interpolation='nearest', vmin = pmin, vmax = pmax, cmap = 'jet')
		plt.title("Residual")
		#	plt.colorbar()
		plt.savefig('../../temp/' + 'Gfit.png')
		plt.close()
	# print('FWHM (arcmin): ', popt[-1]/f2s)
	
	return popt,pcov,Gaussian(MESH1,MESH2,*popt)

def iGauss(stamp, pix_res, high_fwhm, beam_fwhm, verbose = True, return_inverse = False):		###Returns iterative Guassian filter to be used??
	i = 0
	t = 0
	f2s = 1 / (2 * np.sqrt(np.log(4)))	
	high_sig = f2s * high_fwhm / pix_res
	to_filter = stamp
	if verbose:
		print("	Result Std | Fit (peak, sig)		| Peak Fit S/N	| Sig Fit S/N	|")
	while i == 0:
		t += 1
		filter = spi.gaussian_filter(to_filter, sigma = high_sig)
		popt, pcov, fit = GFit(to_filter - filter, pix_res, beam_fwhm, False, 1.5*beam_fwhm)
		std = noise(to_filter-filter, pix_res, 2*beam_fwhm, (len(stamp)-1)/2*pix_res-2*beam_fwhm)[0]*10**-6
		if verbose:
			print(std, popt, np.round(abs(popt[0]/pcov[0,0]), 2), np.round(popt[1]/pcov[1,1], 2))
		if (abs(popt[0]) < std) or (popt[1]/pcov[1,1] < 10):
			i = 1
		else:
			to_filter = to_filter - fit
	if verbose:
		print("Number of iterations: ", t)
	if return_inverse:
		return to_filter
	else:
		return filter

def SingleStackPlot_Show(Image,NSIDE,search_arcmin,circlearc,subtitle,K_r,y):		###Shows a single plot for analysis
	micro = 10**-6
	minK = K_r[0]
	maxK = K_r[1]
	DIV = 5 #Resolution divisor to adjust pixel size
	dAarc =2048/DIV/NSIDE #Arcmins
	RAD = search_arcmin/2
	BASE = np.arange(-RAD, RAD + 2*dAarc, dAarc) #Base array for stack mesh creation
	DR = len(BASE)
	BASE[int((DR)/2-1)] = 0
	MESH1, MESH2 = np.meshgrid(BASE - dAarc/2, BASE - dAarc/2)
	fig, axs = plt.subplots(1, 1, sharex=False, sharey=True, figsize=(6,6))
	min=(np.min(Image)/micro)
	max=(np.max(Image)/micro)
	if K_r == [0,0]:
		mp = axs.pcolormesh(MESH1, MESH2, Image/micro, cmap='jet', vmin=min,vmax=max)
	else:
		mp = axs.pcolormesh(MESH1, MESH2, Image/micro, cmap = 'jet', vmin = minK, vmax = maxK)

	for c in range(len(circlearc)):
		circle1 = plt.Circle((0, 0), circlearc[c], fill = False, color = 'k', ls='--', lw=0.5)
		axs.add_patch(circle1)
	axs.set(aspect='equal')
	axs.title.set_text(subtitle)
	axs.set_xlabel(r'$\Delta\phi$ (arcmin)')
	axs.invert_xaxis()
	axs.invert_yaxis()
	axs.set_ylabel(r'$\Delta\theta$ (arcmin)')
	if y == True:
		fig.colorbar(mp, ax=axs, label = r'$u$y (compton)', shrink=0.8)
	else:
		fig.colorbar(mp, ax=axs, label = r"$uK_{CMB}$", shrink=0.8)
	plt.show()
	plt.close()
	#	print('Plot saved as: ' + savefig)
	return None

def MultiStackPlot(values, NSIDE, search_arcmin, circlearc, subtitles, outname, outloc, K_r, y): #Where K_r -> [min, max] is colorbar range (use [0,0] for whatever cbar range) in default microKelvin(T_CMB) for y=False, else y=True for compton y
	vlen = len(values)
	micro = 10**-6
	minK = K_r[0]
	maxK = K_r[1]
	DIV = 5 #Resolution divisor to adjust pixel size
	dAarc =2048/DIV/NSIDE #Arcmins
	RAD = search_arcmin/2
	BASE = np.arange(-RAD, RAD + 2*dAarc, dAarc) #Base array for stack mesh creation
	DR = len(BASE)
	BASE[int((DR)/2-1)] = 0
	MESH1, MESH2 = np.meshgrid(BASE - dAarc/2, BASE - dAarc/2)
	fig, axs = plt.subplots(1, vlen, sharex=False, sharey=True, figsize=(14,4))
	mins=[]
	maxs=[]
	for v in range(vlen):
		mins.append(np.min(values[v])/micro)
		maxs.append(np.max(values[v])/micro)
	for v in range(vlen):
		if K_r == [0,0]:
			mp = axs[v].pcolormesh(MESH1, MESH2, values[v]/micro, cmap='jet', vmin=np.min(mins),vmax=np.max(maxs))
		else:
			mp = axs[v].pcolormesh(MESH1, MESH2, values[v]/micro, cmap = 'jet', vmin = minK, vmax = maxK)

		for c in range(len(circlearc)):
			circle1 = plt.Circle((0, 0), circlearc[c], fill = False, color = 'k', ls='--', lw=0.5)
			axs[v].add_patch(circle1)
		axs[v].set(aspect='equal')
		axs[v].title.set_text(subtitles[v])
		axs[v].set_xlabel(r'$\Delta\phi$ (arcmin)')
	for v in range(vlen):
		axs[0].invert_yaxis()	
	axs[0].invert_xaxis()
	axs[0].set_ylabel(r'$\Delta\theta$ (arcmin)')
	fig.suptitle(outname)

	if y == True:
		fig.colorbar(mp, ax=axs[:], label = r'$u$y (compton)', shrink=0.8)
	else:
		fig.colorbar(mp, ax=axs[:], label = r"$uK_{CMB}$", shrink=0.8)

	savefig = outloc + outname + '_%.2fArcmin_%.2fpixRes.png' %(search_arcmin, dAarc)
	plt.savefig(savefig, bbox_inches='tight')
	plt.close()
	#	print('Plot saved as: ' + savefig)
	return None

def MultiStackPlot_Show(values, NSIDE, search_arcmin, circlearc, subtitles, K_r, y): #Where K_r -> [min, max] is colorbar range (use [0,0] for whatever cbar range) in default microKelvin(T_CMB) for y=False, else y=True for compton y
	vlen = len(values)
	micro = 10**-6
	minK = K_r[0]
	maxK = K_r[1]
	DIV = 5 #Resolution divisor to adjust pixel size
	dAarc =2048/DIV/NSIDE #Arcmins
	RAD = search_arcmin/2
	BASE = np.arange(-RAD, RAD + 2*dAarc, dAarc) #Base array for stack mesh creation
	DR = len(BASE)
	BASE[int((DR)/2-1)] = 0
	MESH1, MESH2 = np.meshgrid(BASE - dAarc/2, BASE - dAarc/2)
	fig, axs = plt.subplots(1, vlen, sharex=False, sharey=True, figsize=(14,4))
	mins=[]
	maxs=[]
	if vlen==1:
		mins.append(np.min(values)/micro)
		maxs.append(np.max(values)/micro)
		if K_r == [0,0]:
			mp = axs.pcolormesh(MESH1, MESH2, values[0]/micro, cmap='jet', vmin=np.min(mins),vmax=np.max(maxs))
		else:
			mp = axs.pcolormesh(MESH1, MESH2, values[0]/micro, cmap = 'jet', vmin = minK, vmax = maxK)

		for c in range(len(circlearc)):
			circle1 = plt.Circle((0, 0), circlearc[c], fill = False, color = 'k', ls='--', lw=0.5)
			axs.add_patch(circle1)
		axs.set(aspect='equal')
		axs.title.set_text(subtitles[0])
		axs.set_xlabel(r'$\Delta\phi$ (arcmin)')
		axs.invert_xaxis()
		axs.invert_yaxis()
		axs.set_ylabel(r'$\Delta\theta$ (arcmin)')
		if y == True:
			fig.colorbar(mp, ax=axs, label = r'$u$y (compton)', shrink=0.8)
		else:
			fig.colorbar(mp, ax=axs, label = r"$uK_{CMB}$", shrink=0.8)
	else:	
		for v in range(vlen):
			mins.append(np.min(values[v])/micro)
			maxs.append(np.max(values[v])/micro)
		for v in range(vlen):
			if K_r == [0,0]:
				mp = axs[v].pcolormesh(MESH1, MESH2, values[v]/micro, cmap='jet', vmin=np.min(mins),vmax=np.max(maxs))
			else:
				mp = axs[v].pcolormesh(MESH1, MESH2, values[v]/micro, cmap = 'jet', vmin = minK, vmax = maxK)

			for c in range(len(circlearc)):
				circle1 = plt.Circle((0, 0), circlearc[c], fill = False, color = 'k', ls='--', lw=0.5)
				axs[v].add_patch(circle1)
			axs[v].set(aspect='equal')
			axs[v].title.set_text(subtitles[v])
			axs[v].set_xlabel(r'$\Delta\phi$ (arcmin)')
		for v in range(vlen):
			axs[v].invert_xaxis()
		axs[0].invert_yaxis()	
		axs[0].set_ylabel(r'$\Delta\theta$ (arcmin)')

		if y == True:
			fig.colorbar(mp, ax=axs[:], label = r'$u$y (compton)', shrink=0.8)
		else:
			fig.colorbar(mp, ax=axs[:], label = r"$uK_{CMB}$", shrink=0.8)

	plt.show()
	plt.close()
	#	print('Plot saved as: ' + savefig)
	return None

def MultiStackGridPlot(values, search_arcmin, res, circlearc, col_titles,row_titles, oldbeams, newbeam, rebeam=True, meansubtract=True, title=None, savename=None, 
	K_r=[0,0], units=r"$\mu K_{CMB}$", cbar_dec_place=1 , showplt=True, fmt="png",dpi=400, rasterized_plot=True):
	'''Where values must be a <=2d list or array of equal-sized 2d image arrays. Each row must have the same length.
	Oldbeams must be same shape as values, newbeam is single fwhm.
	K_r -> [min, max] is colorbar range (use [0,0] for whatever cbar range)'''
	values=np.asarray(values)
	vshape=values.shape
	print(vshape)
	if len(vshape)==3:	###i.e. 1d list/array of images (2d)
		vshape=(1,vshape[0])
	
	# micro = 10**-6
	# DIV = 5 #Resolution divisor to adjust pixel size
	# dAarc =2048/DIV/NSIDE #Arcmins
	RAD = search_arcmin/2
	BASE = np.arange(-RAD, RAD + 2*res, res) #Base array for stack mesh creation
	DR = len(BASE)
	BASE[int((DR)/2-1)] = 0
	MESH1, MESH2 = np.meshgrid(BASE - res/2, BASE - res/2)
	print(vshape)
	factor=2
	gridxscale=factor*vshape[1]+factor
	gridyscale=factor*vshape[0]+1
	print(vshape,gridxscale,gridyscale)
	
	fig, axs = plt.subplots(vshape[0], vshape[1], figsize=(gridxscale,gridyscale),sharex='col',sharey='row')
	if vshape[0]==1:
		print(vshape[0])
		axs=np.reshape(axs,(1,vshape[1]))
	print(axs.shape)
	mins=[]
	maxs=[]
	oldbeams=np.asarray(oldbeams)
	print('oldbeams: ',oldbeams)
	if rebeam:
		for i in range(vshape[0]):
			for j in range(vshape[1]):
				values[i,j] = change_stamp_gaussian_beam(values[i,j],res,oldbeams[i,j],newbeam)

	###Setting colorbar range
	for i in range(vshape[0]):
		for j in range(vshape[1]):
			if meansubtract:
				values[i,j]-=np.mean(values[i,j])
			mins.append(np.min(values[i,j]))
			maxs.append(np.max(values[i,j]))
	if K_r==[0,0]:
		minK=np.min(mins)
		maxK=np.max(maxs)
	else:
		minK = K_r[0]
		maxK = K_r[1]

	for i in range(vshape[0]):
		for j in range(vshape[1]):
			# if i!=vshape[1]-1:		###Finding all rows above the bottom row, to share x with
			# 	axs[i,j].sharex(axs[-1,j])
			# else:	###Since RA goes from right to left
			if i==vshape[0]-1:
				axs[i,j].invert_xaxis()
				axs[i,j].set_xlabel(r'$\Delta\phi$ [arcmin]')
			# if j!=0:	###Finding columns not on column #0 to share y with
			# 	axs[i,j].sharey(axs[i,0])
			# else:	###Since DEC goes from top down
			if j==0:
				axs[i,j].invert_yaxis()
				axs[i,j].set_ylabel(r'$\Delta\theta$ [arcmin]')

			###Plotting each, with option for colorbar
			print(i,j,np.min(values[i,j]),np.max(values[i,j]))
			mp = axs[i,j].pcolormesh(MESH1, MESH2, values[i,j], cmap = 'jet', vmin = minK, vmax = maxK, rasterized=rasterized_plot)

			for c in range(len(circlearc)):
				circle1 = plt.Circle((0, 0), circlearc[c], fill = False, color = 'k', ls='--', lw=0.5)
				axs[i,j].add_patch(circle1)
			axs[i,j].set(aspect='equal')
			axs[0,j].set_title(col_titles[j])
		axs[i,0].text(-0.65*vshape[0]/(factor+1),0.5,row_titles[i],verticalalignment='center',size=12,rotation=90,transform=axs[i,0].transAxes)
		
	fig.suptitle(title)
	plt.subplots_adjust(left=(factor+1)*0.24/gridxscale,bottom=(factor+1)*0.3/gridxscale,hspace=(factor+1)*0.22/gridyscale,wspace=(factor+1)*0.1/gridxscale)
	cbar_ax = fig.add_axes([0.885+(factor+1)*0.06/gridxscale, 0.18, (factor+1)*0.04/gridyscale, 0.6])
	fig.colorbar(mp, ax=axs[:,:],cax=cbar_ax,format="%."+"%if"%cbar_dec_place)
	cbar_ax.set_title(units)

	###minmax check
	print('mins: ',mins)
	print('maxes: ',maxs)
	if showplt:
		# plt.tight_layout()
		plt.show()
		plt.close()
	else:
		# savefig = savename+"_%.2fArcmin_%.2fpixRes.%s"%(search_arcmin,res,fmt)
		savefig = savename+".%s"%(fmt)
		plt.savefig(savefig,format=fmt,dpi=dpi,bbox_inches="tight")
		plt.close()
		print('Plot saved as: ' + savefig)
	return None

def corr_plot(corr,xset,xskip=4,xdec=2,axis_label=None,title=None,shift=0.,custom_cmap=None,cbar_lims=[-1,1],cbar_ticks=[-1,-.5,0,.5,1],savename=None,saveformat="pdf",fsize=14,smaller_fsize=11,rasterized_plot=True):
	# normi = mpl.colors.Normalize(vmin=cbar_lims[0], vmax=cbar_lims[1])
	plt.rc('xtick', labelsize=smaller_fsize)
	plt.rc('ytick', labelsize=smaller_fsize)
	plt.minorticks_on()
	'''xskip==int, for number of ticks to skip per label, xdec==int for number of decimal places'''
	if custom_cmap:
		plt.imshow(corr,origin='lower',cmap=custom_cmap,vmin=cbar_lims[0],vmax=cbar_lims[1],rasterized=rasterized_plot)#cmap=plt.cm.get_cmap('afmhot').reversed())
	else:
		plt.imshow(corr,origin='lower',cmap=plt.cm.get_cmap('RdYlBu').reversed(),vmin=cbar_lims[0],vmax=cbar_lims[1],rasterized=rasterized_plot)#cmap=plt.cm.get_cmap('afmhot').reversed())
	if xdec==0:
		print(np.array(np.round(xset[::xskip],xdec),dtype=int))
		plt.xticks(np.arange(len(xset))[::xskip]-shift,labels=np.array(np.round(xset[::xskip],xdec),dtype=int))
		plt.yticks(np.arange(len(xset))[::xskip]-shift,labels=np.array(np.round(xset[::xskip],xdec),dtype=int))
	else:
		plt.xticks(np.arange(len(xset))[::xskip]-shift,labels=np.round(xset[::xskip],xdec))
		plt.yticks(np.arange(len(xset))[::xskip]-shift,labels=np.round(xset[::xskip],xdec))
	if axis_label:
		plt.xlabel(axis_label,fontsize=fsize)
		plt.ylabel(axis_label,fontsize=fsize)
	# p.set_norm(normi)
	plt.colorbar(ticks=cbar_ticks)
	if title:
		plt.title(title,fontsize=fsize)
	if savename:
		plt.savefig(savename+".%s"%saveformat,dpi=400,format=saveformat,bbox_inches="tight")
		print("Plot saved as: %s.%s"%(savename,saveformat))
	else:
		plt.tight_layout()
		plt.show()
	plt.close()
	return None

##################General Fitting##################
####My new generalized fitting procedure....
class func_param:
	def __init__(self,func_name: str = "",text_name: str = r"",fit_range: int or float or np.ndarray = np.arange(0,1.01,.1), prior_weight: None or np.ndarray = None, is_linear : bool = False):
		'''Class to setup a function parameter attributes needed to pass through to funcFit or other similar use...

		Parameters
		----------
		func_name : str (Default = "")
			What the function's parameter is called (kwarg) to pass through to the function.
		
		text_name : str (Default = r"")
			The parameter's official name once printed on a plot or in-text (i.e. LaTeX format).
			This will be used for any plots, tables, etc.
		
		fit_range : int, float, or array (Default = np.arange(0,1.01,0.1) == array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]))
			Range of desired values to fit for (When requiring probability density calculations such as BE).

			If int, float, or array with len==1, it will be considered a constant and simply passed onto the fitting function as such (not fit for)...
			For GLS, fit_range only dictates if it should be considered a constant (if int or float) or free-fit (if array with len > 1), 
			since fit can be determined exactly without generating the probability density.
		
		prior_weight : None or array  (Default = None)
			Use if fitting is wanted with any priors applied.  I.e. weighting certain fit values more than others.
			Should be same size as fit_range, if fit_range is a constant, prior_weight has no influence on the final output.

		is_linear : bool (Default = False)
			For if you wish to identify when a parameter is linear, simplifying BE fitting (GLS is where you already assume it is linear)

		Returns
		-------

		Function parameter class object with the above defined attributes, to be used in function fitting processes.

		'''


		self.func_name=func_name
		self.text_name=text_name
		self.fit_range=fit_range
		self.prior_weight=prior_weight
		self.has_prior=False
		self.is_linear=is_linear
		if isinstance(fit_range,int) or isinstance(fit_range,float):
			self.constant=True
		elif len(fit_range)==1:
			warnings.warn("%s: Length of fit_range is 1, assuming it is a constant and changing it to a float..."%func_name)
			self.fit_range=float(fit_range[0])
			self.constant=True
		else:
			self.constant=False
			if prior_weight is not None:
				if fit_range.shape!=prior_weight.shape:
					raise ValueError("%s: A prior weight was assigned but is not the same shape as the provided fit_range..."%func_name)
				else:
					self.has_prior=True
	pass


def prob_level(prob_matrix,percentile):
	'''Given a probability density matrix, returns the level containing the desired percentile ABOVE that level (0<percentile<1).  I.e. All probabilities >=returned level correspond to that interval.
	This assumes the probability matrix is considered roughly continuous and similar to a normal distribution (only one definite peak).'''
	flattened=prob_matrix.flatten()
	flattened.sort()
	csum=np.cumsum(flattened)/np.sum(flattened)
	return np.min(flattened[csum>=(1-percentile)])

def CI_edges(prob1D,prob_range,CI):
	'''Given a 1D probability density, returns bounds [lower, upper] of **2-tail** Confidence/Credible Interval containing the CI percentile amount.'''
	csum=np.cumsum(prob1D)/np.sum(prob1D)
	return [np.min(prob_range[csum>=(1-CI)/2]), np.min(prob_range[csum>=(1+CI)/2])]		###has to be min bc arrays must be of size>0

def gaussian_range_prior(mean,sigma,step_resolution,sigma_range=3):
	'''returns a fit_range and fit_prior_weight about a desired mean/center, and out to 3 sigma (Default sigma_range=3)'''
	half=np.arange(0,sigma_range*sigma,step_resolution)
	fit_range=np.concatenate([-half[:0:-1],half])+mean
	prior_weight=np.exp(-((fit_range-mean)/sigma)**2/2)		###Normalize term omitted since it will just be normalized later (not affecting fitting result)
	return fit_range, prior_weight

def sigma_levels_nD_Gaussian(n,s):
	'''Calculates percentage containing s-sigma of a n-dimensional (radially-symmetric) gaussian'''
	return spsp.gammainc(n/2,s**2/2)

def funcFit(func_toFit : Callable, x_values : np.ndarray or list, y_values : np.ndarray, func_params: list = [func_param("var_name",r"text$_{name}$")], variance: None or np.ndarray =None, 
method: str = "BE", best_fit : str = "median", absolute_variance : bool = True, MemoryOverride : bool = False, CI_method: str = "standard", showBnds_1D : list or np.ndarray = [1], showFitValues : bool = True, showBnds_2D : list or np.ndarray = [1,2,3],
cornerplot: bool = True, cmap : mpl.colors.Colormap = mpl.colors.LinearSegmentedColormap.from_list("",["white","lightblue","royalblue","navy"]), cmap_sigma_max: float = 4,  cornerplot_savename : str = None, save_format : str = "png", n_d_bnds : list = [], return_marg_densities: np.ndarray or list = None, font_size : float = 12, **func_toFit_kwargs):
	"""Using BE (Bayesian Estimation, Default) or GLS (Generalized Least Squares) to fit given data to the desired function.
	
	Parameters
	----------
	func_toFit : Callable function(x_values,func_params) : 
		Desired function to fit.  
		Requires a set of independent x_values and func_params parameter objects to pass through func_toFit.  
		Should be capable of numpy array scaling.
	
	x_values : list, array, or list of arrays (<- emphasis on LIST) : 
		Independent variable(s). 
		
		Will be converted to np.ndarray by stacking along the axis=-1 to preserve numpy vectorization??...
	
	y_values : list or array : 
		Dependent variables(s)/measurements.  Should be same lenth as x_values. 
		
		Will be converted to np.ndarray with np.asarray()...
	
	func_params : list of func_param(func_name, text_name, fit_range, and prior_weight) objects : 
		func_name == kwarg name to be passed through func_toFit
		
		text_name == raw string name to be pass through all plots, etc.  i.e. r"2 S$_\nu$"
		
		fit_range == set of all fit values to attempt to fit for.  If constant, provide as int, float, or len==1 array (or pass as **func_toFit_kwargs).  
		Array only used in generating BE probability density.  GLS only looks at if non-constant (array of len>1)
		
		prior_weight == If any prior is desired, must be same size as fit_range.  Fit with method="GLS" does not currently take any priors into account.

		Any func_param assigned a fit_range of int, float, or 1D array with len==1, will be interpreted as a desired constant (not fit for)
		For method="BE" (Default), each fitting parameter's fit_range must be assigned a 1D array (len>1) covering the desired values to generate the probabilty density over.
		For method="GLS", each fitting parameter's fit_range must be assigned a 1D array (len>1) to be considered non-constant.  If non-constant (np.ndarray w/ len>1), actual fit_range values do not matter (GLS does not generate probability densities)
	
	variance : None, float, 1D array, or 2D array (covariance matrix) : 
		(Default) None == All variances are assumed equal-weighted.
		
		Float == All variances assumed same provided float value.  
		
		1D array == Individual variances for each measurement, assumed independent from each other.
		
		2D array (Covariance Matrix) == Full covariance matrix, equivalent to 1D if off-diagonals are all zero (further equivalent to float if all same).

	method : str, ["BE","GLS"] : 
		(Default) "BE" == Bayesian Estimation.  Valid for non-linear fits (careful defining ranges for func_toFitParams which can greatly impact fit & accuracy)
		
		"GLS" == Generalized Least Squares.  Valid for linear fits (Regression), often much faster since probability densities do not have to be generated.  Priors are not considered.
	
	best_fit : str, ["median","average","peak1d","peak"] : 
		(Default) "median"== Best fit reported is median of 1D parameter densities (50th percentile).  Commonly used in astronomy & academic results.  Median will also be reported if unrecognized value is passed.
		"average" == Best fit reported is mean of 1D parameter density distribution.
		"peak1d" == Highest probability of 1D distribution.
		"peak" == Highest probability of p-dimensional distribution (for p total parameters fit).

	absolute_variance : bool : 
		(Default) True == Variances passed represents the variance in absolute sense
		
		False == Variances passed are only relative magnitudes.  Final reported covariance matrix is scaled by the sample variance of the residuals of the fit
	
	MemoryOverride : bool = False : 
		If you want to bypass a warning inplace to prevent BE from creating large arrays over ~4 GB in size.

	CI_method: str = "standard" : 
	Confidence Interval method for 1D bounds.  (Default) "standard" == quantile selection where lower and upper bounds exlude an equal percentage. 
	Else, calculates most probable marginalized densities first (not recommended, but was old way).

	showBnds_1D : list or ndarray = [1] : 
		(Default) [1] (show only 1 sigma bounds)
		Can select [1,2,3] corresponding to 1, 2, or 3 sigma 1D bounds.
	
	showFitValues : bool = True : 
	Show best fit values and associated (1-sigma) errors above diagonals of corner plot.  1-sigma errors are currently only option to show as of now.

	showBnds_2D : list or ndarray = [1,2,3] : 
		(Default) [1,2,3] corresponding to 1, 2, or 3 sigma, 2D bounds.
		Show sigma bounds on 2D contour, can select any set of sigma bounds, but need to be monatonically increasing. 

	cornerplot : bool = True: 
		(Default) True == Generate and show corner plot of parameter posterior distributions (For method="BE" only)

	cmap : mpl.colors.Colormap = mpl.colors.LinearSegmentedColormap.from_list("",["white","lightblue","royalblue","navy"]) : 
	Colormap to use in the corner plot for the 2D contours and any 1D bound markers.  Selection is set to cmap(0.9-(b-1)/(len(showBnds_2D)))

	cmap_sigma_max : float = 4 : 
	cmap selection is linear, thus need to determine max sigma to classify as 0 (default==white) color

	cornerplot_savename : str = "" : 
		(Default) "", resulting in the plot showing and not saving.  Any other string will result in it being saved.
	
	save_format : str  = "png": 
		(Default) "png", resulting in all plots being saved as save_format, also requires savenames to be passed.
	
	n_d_bnds : list = [] : 
	List of sigma bounds to return all possible (n #) parameter combinations within.  Corresponds to n-dimensional gaussian bounds, used in func_fitPlot.
	
	return_marg_densities: np.ndarray or list = None :
	list (or list of lists) or array of desired densities to return marginalized over.  
	Example: For func_params (to fit) "a","b", and "c" , a return_marg_densities=[["a"],["a","b"]] will return a dictionary of:
	A 2-d prob density array marginalized over parameter "a", and a 1-d prob density array corresponding to "c" (marginalized over "a" and "b").

	font_size : float = 12 :
	Font size for all text on corner plot axes.  Diagonal best fit is displayed 2 pts smaller (to ensure it fits)
	
	**func_toFit_kwargs : 
	Any additional kwargs needed passed through func_toFit.

	"""
	# print("----FuncFit----")
	###SETUP
	warnings.filterwarnings("error",category=RuntimeWarning)
	#Just in case formatting
	x_values=np.stack(x_values,axis=-1)		###We want to ensure vectorization is preserved by stacking along the last dimension
	# print(x_values.shape)
	y_values=np.asarray(y_values)
	N=len(y_values)
	# print(N)
	#Sorting parameters into constants and non-constants
	const_func_params=[p for p in func_params if p.constant]
	const_func_kwargs=dict([[p.func_name,p.fit_range] for p in const_func_params])	#dictionary of constant kwargs to pass through function
	# print("Constant Parameters: ",const_func_kwargs)
	tofit_func_params=[p for p in func_params if not p.constant]
	# print("Variable/to-Fit Parameter Function Names: ",[p.func_name for p in tofit_func_params])
	fit_param_len=len(tofit_func_params)
	fit_param_shape=[len(p.fit_range) for p in tofit_func_params]

	###Inverse covariance matrix
	if variance is None:	###Then assume all y values are evenly weighted
		inv_cov=np.identity(N)
	elif np.prod(np.asarray(variance).shape)==1:	###i.e. given one error assumed for all (with zero correlation/off-diagonals)
		inv_cov=np.identity(N)/variance
	elif len(np.asarray(variance).shape)==1:		###i.e. given y errors, assumes off-diagonals are zero (i.e. no covariance between y measurements)
		if len(np.asarray(variance))!=N:
			raise ValueError("1D variance given but is not the same size as provided x-values...")
		inv_cov=np.identity(N)
		for i in range(N):
			inv_cov[i,i]=variance[i]**(-1)		###Filling diagonals
	else:		###Assuming now a full covariance matrix was given
		inv_cov=np.linalg.inv(variance)
	# print(inv_cov)
	###Let's do the GLS version first:
	if method.upper() == "GLS":
		print("-----GLS Fitting-----")
		##Gotta generate the linear X matrix terms (what the parameters are multipled by)
		#Create a dictionary of kwargs from identity matrix for parameter factors
		tofit_iden=np.identity(fit_param_len)
		tofit_iden_func_kwargs=dict([[tofit_func_params[p].func_name,tofit_iden[p][None,...]] for p in range(fit_param_len)])	#dictionary of constant kwargs to pass through function
		# print(tofit_iden_func_kwargs)
		x_matrix=func_toFit(x_values[...,None],**const_func_kwargs,**tofit_iden_func_kwargs,**func_toFit_kwargs)	###X-matrix, shape of (y_values len,fit_func_params len), passing all constants as given in case any were nonlinear
		const_offset=func_toFit(x_values,**const_func_kwargs,**dict([[p.func_name,0] for p in tofit_func_params]),**func_toFit_kwargs)
		###In case any of the constants were nonlinear, we now subtract our constant term from the X matrix (removing the columns of 1s that would otherwise exist in the X matrix)
		x_matrix=x_matrix-np.expand_dims(const_offset,1)		
		###Similarly, we subtract the constant term from the measured y_vector.  Any constant term will have now been accurately removed from the array.  If fitting of it was instead desired, designate it in the func_toFit and as a fit_param
		y_vector=(y_values-const_offset).T

		# print("Constant offset: ",const_offset,const_offset.shape,x_matrix.shape)
		# print((y_values-const_offset).shape,y_vector.shape,x_matrix.shape)
		# y_vector=(y_values[...,]).T

		###The actual fit		
		best_fit_params=np.linalg.inv(x_matrix.T@inv_cov@x_matrix)@x_matrix.T@inv_cov@y_vector
		# print(fit)

		scale=1
		if not absolute_variance:
			scale=(y_vector-x_matrix@best_fit_params).T@(y_vector-x_matrix@best_fit_params)/(N-len(best_fit_params))
			# print("Scaled By: ",scale)
		fit_param_cov=np.linalg.inv(x_matrix.T@inv_cov@x_matrix)*scale


	###Now the BE method:
	elif method.upper() == "BE":
		print("-----BE Fitting-----")
		fit_ranges=[p.fit_range for p in tofit_func_params]	###All ranges to fit
		nonlinear_tofit_func_params=[p for p in tofit_func_params if not p.is_linear]		###All non-linear parameters to fit
		nonlinear_fit_ranges=[p.fit_range for p in nonlinear_tofit_func_params]				###All non-linear ranges to fit
		# nonlinear_ref=np.asarray([p for p in range(fit_param_len) if not tofit_func_params[p].is_linear])		###For creating dimensions of grid array... don't actually use this term
		nonlinear_len=len(nonlinear_fit_ranges)
		linear_tofit_func_params=[p for p in tofit_func_params if p.is_linear]				###All linear parameters to fit
		linear_fit_ranges=[p.fit_range for p in linear_tofit_func_params]					###All linear ranges to fit
		linear_ref=np.asarray([p for p in range(fit_param_len) if tofit_func_params[p].is_linear])		###For creating dimensions of grid array
		# print(linear_ref)
		# print("CP 1")
		prior_check=[p.has_prior for p in tofit_func_params]
		###Checking array size needed
		max_param_size=np.max([fit_param_len,N])*np.prod([len(r) for r in fit_ranges])
		# print(max_param_size)		###Check
		if max_param_size*64/8>(4*10**9) and not MemoryOverride:
			raise MemoryError("Expected memory use (%.2f GB) from parameter size is greater than 4 GB.... you may want to reduce your parameters' fit_range. Set MemoryOverride=True to override this caution."%(max_param_size*8/10**9))
		# print("CP 2")
		###Old way
		##Generating all possible combinations
		# fit_param_grid=np.stack(np.meshgrid(*fit_ranges,indexing="ij"),axis=0)	###Stacking the combinations together across row (axis=0)
		# fit_param_grid=fit_param_grid.reshape(fit_param_len,np.prod(fit_param_shape))	###flattening down to 2d array where each column is a combination
		# try:	###When dealing with large arrays... particularly if x_values is also a multidimensional array that max_param_size didn't catch... 
		# 	fit_y_grid=func_toFit(x_values[...,None],**const_func_kwargs,**dict([[tofit_func_params[p].func_name,fit_param_grid[p]] for p in range(fit_param_len)]))	###y of all parameter combinations
		# except:	###Instead opting to try to loop through each parameter combination? IDK if this is effective, if it does work, it'll be rather slow... YEP SUPER SLOW!!!!
		# 	fit_y_grid=[func_toFit(x_values[...,None],**const_func_kwargs,**dict([[tofit_func_params[p].func_name,fit_param_grid[p][i]] for p in range(fit_param_len)])) for i in range(np.prod(fit_param_shape))]	###y of all parameter combinations
		# print("------")
		
		###New Way!!! Now first creating grid only of non-linear terms needed to fit with combination of 0 or 1 for linear terms (finding each linear term per combination)
		##This helps reduce memory usage, if, per example, someone has a 2d x_values that's 100s of elements long... *COUGH*
		linear_tofit_iden=np.identity(len(linear_fit_ranges))
		# print(linear_tofit_iden)
		###Old grid generation way... seems a bit slower and likely more memory intensive than new option
		# nonlinear_fit_param_grid=np.stack(np.meshgrid(*[p.fit_range for p in tofit_func_params if not p.is_linear],indexing="ij"),axis=0)
		# nonlin_grid_shape=nonlinear_fit_param_grid.shape
		# nonlinear_fit_param_grid=nonlinear_fit_param_grid.reshape(len(nonlinear_fit_ranges),np.prod([len(r) for r in nonlinear_fit_ranges]))
		# constant_offset=func_toFit(x_values[...,None],**const_func_kwargs,**dict([[nonlinear_tofit_func_params[p].func_name,nonlinear_fit_param_grid[p]] for p in range(len(nonlinear_fit_ranges))]),
		constant_offset=func_toFit(np.expand_dims(x_values,axis=tuple(np.arange(nonlinear_len)+len(x_values.shape))),**const_func_kwargs,**dict([[nonlinear_tofit_func_params[p].func_name,np.expand_dims(nonlinear_fit_ranges[p],axis=tuple(np.delete(np.arange(nonlinear_len),p)))] for p in range(nonlinear_len)]),
		**dict([[p.func_name,0] for p in linear_tofit_func_params]),**func_toFit_kwargs)	###Holding all linear terms to 0 to find constant independent term
		# print("CP 3")
		# linear_iden_terms=[func_toFit(x_values[...,None],**const_func_kwargs,**dict([[nonlinear_tofit_func_params[p].func_name,nonlinear_fit_param_grid[p]] for p in range(len(nonlinear_fit_ranges))]),
		linear_iden_terms=[func_toFit(np.expand_dims(x_values,axis=tuple(np.arange(nonlinear_len)+len(x_values.shape))),**const_func_kwargs,**dict([[nonlinear_tofit_func_params[p].func_name,np.expand_dims(nonlinear_fit_ranges[p],axis=tuple(np.delete(np.arange(nonlinear_len),p)))] for p in range(nonlinear_len)]),
		**dict([[linear_tofit_func_params[p].func_name,linear_tofit_iden[i,p]] for p in range(len(linear_tofit_func_params))]),**func_toFit_kwargs) for i in range(len(linear_fit_ranges))]
		# print("CP 4")
		nonlinear_fit_param_grid=None; del nonlinear_fit_param_grid		###To try to save memory? Hopefully? (Idk)
		###Shaping back into grid, expand out axes for linear terms, multiply each component though, add together, flatten back out... so simple!!!
		# fit_y_grid=np.expand_dims(constant_offset.reshape((len(y_values),*nonlin_grid_shape[1:])),tuple(linear_ref+1))
		fit_y_grid=np.expand_dims(constant_offset,tuple(linear_ref+1))
		constant_offset=None; del constant_offset
		# print("CP 5")
		###Newer bestter way?? (by not actually saving linear_terms prior to summing), cuts time and memory by 1-4% in one example basic test...
		fit_y_grid_copy=np.copy(fit_y_grid)		###The constant term to be subtracted from each linear term that was calculated
		for i in range(len(linear_fit_ranges)):
			# fit_y_grid=fit_y_grid+(np.expand_dims(linear_iden_terms[i].reshape((len(y_values),*nonlin_grid_shape[1:])),tuple(linear_ref+1))-fit_y_grid_copy)*np.expand_dims(linear_fit_ranges[i],[0]+list(np.delete(np.arange(fit_param_len)+1,linear_ref[i])))
			fit_y_grid=fit_y_grid+(np.expand_dims(linear_iden_terms[i],tuple(linear_ref+1))-fit_y_grid_copy)*np.expand_dims(linear_fit_ranges[i],[0]+list(np.delete(np.arange(fit_param_len)+1,linear_ref[i])))
		# print("CP 6")
		###Old stuff
		# linear_terms=[(np.expand_dims(linear_iden_terms[i].reshape((len(y_values),*nonlin_grid_shape[1:])),tuple(linear_ref+1))-fit_y_grid)*np.expand_dims(linear_fit_ranges[i],[0]+list(np.delete(np.arange(fit_param_len)+1,linear_ref[i]))) for i in range(len(linear_fit_ranges))]
		# for l in linear_terms:	###Tried to do this a fancy way with np.add.reduce().... this is probabily the easiest... I still don't like it bc of potential memory concerns
		# 	fit_y_grid=fit_y_grid+l
		# fit_y_grid=fit_y_grid+eval("0"+"".join(["+linear_terms["+str(i)+"]" for i in range(len(linear_terms))]))		###Super cheesy way... doesn't look time or memory efficient
		fit_y_grid_copy=None;linear_terms=None;constant_offset=None;linear_iden_terms=None; del linear_terms, constant_offset, linear_iden_terms, fit_y_grid_copy
		# print("CP 7")
		fit_y_grid=fit_y_grid.reshape(len(y_values),np.prod(fit_y_grid.shape[1:]))
		residuals=y_values[...,None]-fit_y_grid
		fit_param_grid=None; fit_y_grid=None; del fit_param_grid, fit_y_grid		###Since we shouldn't need these anymore and want to try to save memory
		# print("CP 8")
		prob_density=np.exp(-np.einsum("ij,ji->i",residuals.T,inv_cov@residuals)/2).reshape(fit_param_shape)
		residuals=None; del residuals
		###Apply any priors to our likelihood/probability
		if any(prior_check):		###If any priors were assigned...
			print("At least 1 Prior Weight Provided \n")
			for p in range(fit_param_len):	###cycle through
				if prior_check[p]:			###If a prior...
					prob_density*=np.expand_dims(tofit_func_params[p].prior_weight,tuple(np.delete(np.arange(fit_param_len),p)))	###Expand prior weight out to same dims as prob_density and multiply with

		# print("CP 9")
		###Generating 1D densities for variances and corner plot
		fit_param_cov=np.zeros([fit_param_len,fit_param_len])
		fit_param_1d_densities=[]
		best_fit_params=[]
		fit_param_1d_errorBnds=[]
		for p in range(fit_param_len):
			###marginalizing over parameters to form 1D densities
			p_1d_density=np.sum(prob_density,axis=tuple(np.delete(np.arange(fit_param_len),p)))		###Summing along all axes except the parameter of interest
			p_1d_density/=np.sum(p_1d_density)			###Normalized
			fit_param_1d_densities.append(p_1d_density)

			###Calculating confidence interval bounds out to 3 sigma in 1D marginalized probabilities
			if CI_method.lower()=="standard":	###Selecting percentiles upper and lower
				fit_param_1d_errorBnds.append([CI_edges(p_1d_density,fit_ranges[p],sigma_levels_nD_Gaussian(1,i+1)) for i in range(3)])	###cycling through 1-3 sigma levels
			else:	###Old way... selecting highest probabilities first til reaching desired interval/level... not standard and doesn't work well with non-normal distributions (when multiple peaks)
				s1=prob_level(p_1d_density,sigma_levels_nD_Gaussian(1,1))		###1 Sigma level
				s2=prob_level(p_1d_density,sigma_levels_nD_Gaussian(1,2))		###2-Sigma level
				s3=prob_level(p_1d_density,sigma_levels_nD_Gaussian(1,3))		###3-Sigma level
				s_levels=[s1,s2,s3]
				errorBnds=[[np.min(fit_ranges[p][p_1d_density>=s]),np.max(fit_ranges[p][p_1d_density>=s])] for s in s_levels]
				fit_param_1d_errorBnds.append(errorBnds)
			
			###Determining the "Best Fit" value of the parameter
			if best_fit.lower()=="peak1d":
				best_fit_params.append(fit_ranges[p][np.argmax(p_1d_density)])
			elif best_fit.lower()=="average":
				best_fit_params.append(np.average(fit_ranges[p],weights=p_1d_density))
			elif best_fit.lower()=="peak":		###Instead determined from master density distribution.... could result in an outlier report as best fit...
					max_loc=np.unravel_index(np.argmax(prob_density),shape=fit_param_shape)
					best_fit_params=[fit_ranges[i][max_loc[i]] for i in range(fit_param_len)]
			else:		###Since throwing an error here would just be a pain, assuming median as the default best_fit choice...
				best_fit_params.append(fit_ranges[p][np.argmin((np.cumsum(p_1d_density)-0.5)**2)])
			

			###Calculating covariance estimate from 1d proabililty (needed to know what we considered the "Best Fit" first tho...)
			fit_param_cov[p,p]=np.sum(p_1d_density*(fit_ranges[p]-best_fit_params[p])**2)	###a weighted covariances around the determined best fit
		fit_param_1d_errorBnds=np.asarray(fit_param_1d_errorBnds)
		# print(fit_param_1d_errorBnds,fit_param_1d_errorBnds.shape)
		# print(best_fit_params)
		# print("CP 10")
		###Generating 2D densities for covariance estimates and corner plot
		fit_param_2d_densities=[]
		fit_param_2d_levels=[]
		for p1 in range(fit_param_len):
			fit_param_2d_densities.append([])
			fit_param_2d_levels.append([])
			for p2 in range(fit_param_len)[:p1]:
				# print(p1,p2)
				p_2d_density=np.sum(prob_density,axis=tuple(np.delete(np.arange(fit_param_len),[p1,p2])))		###Summing along all axes except the TWO parameters of interest
				p_2d_density/=np.sum(p_2d_density)		###Normalizing
				fit_param_2d_densities[p1].append(p_2d_density)

				###Determining the levels for 1, 2, and 3 sigma (the correct way now...)
				temp=[]
				for s in showBnds_2D:
					temp.append(prob_level(p_2d_density,sigma_levels_nD_Gaussian(2,s)))
				# print(temp)
				for t in range(len(temp[1:])):
					if temp[t+1]>=temp[t]:
						temp[t+1]=np.copy(temp[t])/1.000001
				fit_param_2d_levels[p1].append(temp)
				# s1=prob_level(p_2d_density,sigma_levels_nD_Gaussian(2,1))		###1 Sigma level	(2D Gaussian Contour level)
				# s2=prob_level(p_2d_density,sigma_levels_nD_Gaussian(2,2))		###2-Sigma level	(2D Gaussian Contour level)
				# s3=prob_level(p_2d_density,sigma_levels_nD_Gaussian(2,3))		###3-Sigma level	(2D Gaussian Contour level)
				# print(s1,s2,s3)
				###Edge cases when some combination of s1==s2==s3
				# if s2==s1:
				# 	s1*=1.00001
				# if s3==s2:
				# 	s2*=1.000001
				# if any(np.array(temp[1:]))
				# fit_param_2d_levels[p1].append([s1,s2,s3])
				

				###Covariances
				fit_param_cov[p1,p2]=fit_param_cov[p2,p1]=np.sum(p_2d_density*((fit_ranges[p2][:,None]-best_fit_params[p2])*(fit_ranges[p1][None,:]-best_fit_params[p1])))
		# print(fit_param_cov)

		prob_density/=np.max(prob_density)		###Normalize to max
		# print(best_fit_params)	###CHECK
		###Plotting
		##Colors
		# bnd_colors=["lightblue","royalblue","navy"]
		# blue_cmap=mpl.cm.get_cmap("Blues")
		if cornerplot:
			fig,axs=plt.subplots(nrows=fit_param_len,ncols=fit_param_len,sharex='col',figsize=(1+2*fit_param_len,1+2*fit_param_len))
			for i in range(fit_param_len):	###Corner plot rows
				for j in range(fit_param_len):	###Corner plot cols
					# print(i,j)
					if j>i:
						fig.delaxes(axs[i,j])
					elif i==j:		###1D marginalized distributions
						axs[i,j].get_yaxis().set_visible(False)
						ref=tuple(np.concatenate((np.arange(i),np.arange(i+1,fit_param_len))))	###Making axes of chi_sq_probs that I need to sum over (all but i axis)
						# print('ref: ',ref)	###CHECK
						axs[i,j].plot(fit_ranges[i],fit_param_1d_densities[i],marker="",linestyle="-")
						axs[i,j].vlines(best_fit_params[i],0,np.max(fit_param_1d_densities[i])*1.1,color="k",linestyle="--",alpha=0.9)
						###Sigma Bounds
						for b in showBnds_1D:
							axs[i,j].vlines(fit_param_1d_errorBnds[i,b-1,:],0,np.max(fit_param_1d_densities[i])*1.1,color=cmap(0.9-(b-1)/(len(showBnds_2D))),linestyle="--",alpha=0.7+b/40)	###Scaling alpha bc lighter blues are tougher to see
						axs[i,j].set_ylim([0,np.max(fit_param_1d_densities[i])*1.1])
						if showFitValues:		###For 1 sigma values only (since any other would likely be confusing.... maybe add options later)
							if np.max(np.abs(fit_ranges[i]))<0.01:
								try:
									factor=np.ceil(abs(np.log10(np.abs(best_fit_params[i]))))
									# if factor==np.inf:	###Happens when best fit is zero...
								except:
									try:
										factor=np.ceil(abs(np.log10(fit_param_1d_errorBnds[i,0,1]-best_fit_params[i])))		###So instead use factor of highest bound
									except:
										factor=np.ceil(np.abs(np.log10(np.max(fit_ranges[i]))))
										warnings.warn("Best fit %s and 1 Sigma bound are either not clearly defined or overlapping... you might need to readjust your bounds or resolution..."%tofit_func_params[i].text_name,UserWarning)
									# if factor==np.inf:	###ANDDDDD if even the upper bound is also zero... just use max of fit ranges.... chances are resolution needs to be improved....
										
								# print(factor)	###CHECK
								sf=math.ceil(max(0,-np.log10(np.max(fit_ranges[i][1:]-fit_ranges[i][:-1])*10**factor)))		###Sig fig selection for float display
								axs[i,j].set_title(tofit_func_params[i].text_name+r"=$%.*f_{-%.*f}^{+%.*f}\times10^{-%i}$"%(sf,best_fit_params[i]*10**factor,sf,(best_fit_params[i]-fit_param_1d_errorBnds[i,0,0])*10**factor,sf,(fit_param_1d_errorBnds[i,0,1]-best_fit_params[i])*10**factor,factor),size=(font_size-2))
							elif np.max(np.abs(fit_ranges[i]))>1000:
								factor=np.floor(abs(np.log10(np.abs(best_fit_params[i]))))
								# print(factor)	###CHECK
								sf=math.ceil(max(0,-np.log10(np.max(fit_ranges[i][1:]-fit_ranges[i][:-1])/10**factor)))		###Sig fig selection for float display
								axs[i,j].set_title(tofit_func_params[i].text_name+r"=$%.*f_{-%.*f}^{+%.*f}\times10^{%i}$"%(sf,best_fit_params[i]/10**factor,sf,(best_fit_params[i]-fit_param_1d_errorBnds[i,0,0])/10**factor,sf,(fit_param_1d_errorBnds[i,0,1]-best_fit_params[i])/10**factor,factor),size=(font_size-2))
							else:
								sf=math.ceil(max(0,-np.log10(np.max(fit_ranges[i][1:]-fit_ranges[i][:-1]))))		###Sig fig selection for float display
								axs[i,j].set_title(tofit_func_params[i].text_name+r"=$%.*f_{-%.*f}^{+%.*f}$"%(sf,best_fit_params[i],sf,best_fit_params[i]-fit_param_1d_errorBnds[i,0,0],sf,fit_param_1d_errorBnds[i,0,1]-best_fit_params[i]),size=(font_size-2))
						if tofit_func_params[i].has_prior:		###If any prior applied, plot as a dashed line and normalize?
							axs[i,j].plot(fit_ranges[i],tofit_func_params[i].prior_weight/np.max(tofit_func_params[i].prior_weight)*np.max(fit_param_1d_densities[i]),marker="",linestyle="-.",color="grey")		
					elif i>j:
						if j!=0:	###i.e. not leftmost
							axs[i,j].sharey(axs[i,0])
							axs[i,j].tick_params(labelleft=False)	###To not show tick labels on inside plots
						# print([fit_param_2d_levels[i][j][::-1][b-1] for b in showBnds_2D])
						axs[i,j].contourf(*np.meshgrid(fit_ranges[j],fit_ranges[i]),fit_param_2d_densities[i][j].T,levels=[fit_param_2d_levels[i][j][::-1][b] for b in range(len(showBnds_2D))]+[1],colors=[cmap(0.95-0.95*(b-1)/(cmap_sigma_max)) for b in showBnds_2D[::-1]])
					axs[i,j].minorticks_on()
				if np.max(fit_ranges[i])<0.2:		###Have to do this AFTER the plot is generated in order to grab the scientific notation text
					plt.subplots_adjust(wspace=0.3,hspace=0.3)	###General default is 0.2, so increasing slightly to prevent any sci notation overlap
					axs[i,0].ticklabel_format(style="sci",scilimits=(0,0))
					axs[-1,i].ticklabel_format(style="sci",scilimits=(0,0))
					axs[i,0].figure.canvas.draw()
					axs[i,0].yaxis.get_offset_text().set_visible(False)
					axs[-1,i].xaxis.get_offset_text().set_visible(False)
					axs[i,0].set_ylabel(tofit_func_params[i].text_name + "  [" + axs[i,0].get_yaxis().get_offset_text().get_text() + "]",size=font_size)
					axs[-1,i].set_xlabel(tofit_func_params[i].text_name + "  [" + axs[-1,i].get_xaxis().get_offset_text().get_text() + "]",size=font_size)
				elif np.max(fit_ranges[i])>1000:
					plt.subplots_adjust(wspace=0.3,hspace=0.3)	###General default is 0.2, so increasing slightly to prevent any sci notation overlap
					axs[i,0].ticklabel_format(style="sci",scilimits=(0,0))
					axs[-1,i].ticklabel_format(style="sci",scilimits=(0,0))
					axs[i,0].figure.canvas.draw()
					axs[i,0].yaxis.get_offset_text().set_visible(False)
					axs[-1,i].xaxis.get_offset_text().set_visible(False)
					axs[i,0].set_ylabel(tofit_func_params[i].text_name + "  [" + axs[i,0].get_yaxis().get_offset_text().get_text() + "]",size=font_size)
					axs[-1,i].set_xlabel(tofit_func_params[i].text_name + "  [" + axs[-1,i].get_xaxis().get_offset_text().get_text() + "]",size=font_size)

				else:
					axs[i,0].set_ylabel(tofit_func_params[i].text_name,size=font_size)
					axs[-1,i].set_xlabel(tofit_func_params[i].text_name,size=font_size)
			###now ensuring left subplots have same y-axes as bottom subplot's x-axes...
			for i in range(1,fit_param_len):
				axs[i,0].set_yticks(axs[-1,i].get_xticks())
				axs[i,0].set_ylim(axs[-1,i].get_xlim())	###I guess the order matters here?? lim needed after ticks or else ticks will overwrite???? Cool.
			if cornerplot_savename:
				plt.savefig(cornerplot_savename+"."+save_format,format=save_format,dpi=400)
			else:
				plt.show()
			plt.cla()
			plt.clf()
			plt.close("all")	###To save memory
	
	###Generating sigma bounds when applicable
	b_bnds={}
	if  method=="BE" and n_d_bnds:	###Have to find where bounds are (ranges of confidence interval, then determine min and max per x_oversampled)
		for b in n_d_bnds:	###Up to whatever sigma
			s=prob_level(prob_density,sigma_levels_nD_Gaussian(fit_param_len,b))		###b-Sigma level (i.e. 1, 2, 3, etc sigma)
			s_locs=np.where(prob_density>=s)
			# for f in range(fit_param_len):
			# 	print(fit_ranges[f][s_locs[f]])
			# print(len(s_locs),s_locs[0].shape,prob_density.shape)
			param_bnds={tofit_func_params[f].func_name:fit_ranges[f][s_locs[f]] for f in range(fit_param_len)}
			b_bnds[r"%.1f $\sigma$"%b]=param_bnds
			# print(vals.shape,"vals")
			# b_bnds.append([vals.min(axis=1),vals.max(axis=1)])		###Appending min and max of each x_oversampled location
	best_fit_params=dict([[tofit_func_params[f].func_name,best_fit_params[f]] for f in range(fit_param_len)])

	marg_densities={}
	if method=="BE" and (return_marg_densities is not None):	###For when densities are also desired
			return_marg_densities=np.array(return_marg_densities)	##If lists given
			tofit_func_params_keys=[p.func_name for p in tofit_func_params]
			t=True
			if np.prod(return_marg_densities.shape)>0:
				t=type(return_marg_densities[0]) is not list
			if len(return_marg_densities.shape)==1 and t:	###i.e. single density desired
				temp=np.sum(prob_density,axis=tuple(tofit_func_params_keys.index(k) for k in return_marg_densities))
				marg_densities[("Marginalized over: "+", ".join(list(return_marg_densities)))]=temp/np.sum(temp)
			else:
				for r in return_marg_densities:
					temp=np.sum(prob_density,axis=tuple(tofit_func_params_keys.index(k) for k in r))
					marg_densities[("Marginalized over: "+", ".join(list(r)))]=temp/np.sum(temp)
	
	gc.collect()	###To try to keep memory usage in check (for loops, etc)
	if method=="GLS":	###Return Best Fit and covariance
		return best_fit_params, fit_param_cov
	elif method=="BE":	###Return Best Fit, covariance, 1d errorbands for each parameter, and 1,2,3-sigma bounds for full x_oversampled
		fit_param_1d_errorBnds=dict([[tofit_func_params[f].func_name,fit_param_1d_errorBnds[f]] for f in range(fit_param_len)])

		return best_fit_params, fit_param_cov, fit_param_1d_errorBnds, b_bnds, marg_densities


###Since funcFit is getting too cumbersome...
def plot_funcFit(func_toPlot : Callable, x_values : list or np.ndarray, y_values : list or np.ndarray, yerrs: list or np.ndarray = None, x_showFit_vals : None or list or np.ndarray = None, val_label : str = "Data", axis_labels : list or np.ndarray = [r"x",r"y"], 
best_fit_params: dict = None, fit_param_combos : dict = None, linear_paramNames : list or np.ndarray = None, param_scale: dict = None, xlims : list or tuple = [], ylims : list or tuple = [], xscale : str = None, yscale : str = None, grid : bool = True, 
cmap : mpl.colors.Colormap = mpl.colors.LinearSegmentedColormap.from_list("",["white","lightblue","royalblue","navy"]), cmap_sigma_max : float = 4, discrete_colors : list or np.ndarray = None, save_name : str = None, save_format : str = "png", continue_plot : bool = False, font_size : float = 12, **func_toplot_kwargs):
	'''Takes b_bnds output from funcFit and plots sigma bounds along with any provided data.
	
	Parameters
	--------------
	func_toPlot : Callable : 
	Function to plot best fit and sigma bounds

	x_values : list or np.ndarray  : 
	1D locations to plot, may be different than x_values that would be passed to func in fitting process

	y_values : list or np.ndarray : 
	1D array, measured values to plot

	yerrs: list or np.ndarray = None : 
	yerrs to be passed directly through to plt.errorbar(), can be 1D list/array or 2D with a row for min and max (respectively)... 
	see matplotlib.pyplot.errorbar for more specifics.

	x_showFit_vals : None or list or np.ndarray : 
	x_values to be used for plotting best fit and sigma bounds, will be passed through func_toPlot.  
	If None, it will be generated from 0.9 to 1.1 times min and max of x_values (respectively) for best fit (and sigma bounds if fit_param_combos is still given).

	val_label : str = "Data" : 
	Label for data points in plot legend.

	axis_labels : list or np.ndarray = [r"x",r"y"] : 
	Labels for plot axes.

	best_fit_params: dict = None : 
	Dictionary of best fit param from funFit output[0].

	fit_param_combos : dict = None : 
	Nested dictionary of fit param combinations that make up sigma bounds, from funcFit output[3].

	linear_paramNames : list or np.ndarray = None : 
	1D list or array to specify any parameters that are linear (such as those defined as is_linear=True in func_param class).  Should be func_name that will be passed to func_toPlot.

	param_scale: dict = None : 
	Dictionary in the event of any parameters need scaling (such as by some factor of 10) to match data points.  Keys should be same func_name as fit params.

	xlims : list or tuple = [] : 
	Plot x-limits to pass through plt.xlims()

	ylims : list or tuple = [] : 
	Plot y-limits to pass through plt.ylims()

	xscale : str = None : 
	String to pass through plt.xscale(), such as "log" or "symlog"
	
	yscale : str = None : 
	String to pass through plt.yscale(), such as "log" or "symlog"

	grid : bool = True : 
	Show plot grid

	cmap : mpl.colors.Colormap = mpl.colors.LinearSegmentedColormap.from_list("",["white","lightblue","royalblue","navy"])  : 
	Colormap to pass for defining different sigma bounds.  Selection is set to: cmap(0.9-(float(list(fit_param_combos.keys()))-1)/(len(fit_param_combos)))

	cmap_sigma_max : float = 4 : 
	cmap selection is linear, thus need to determine max sigma to classify as 0 (default==white) color

	discrete_colors : list or np.ndarray = None : 
	1D list or array of strings, defining discrete colors of sigma bounds, should be at least as long as number of sigma bounds in fit_param_combos.
	Note: Data points (from x_values and y_values) and best_fit will still use the color of cmap(1.0).
	
	save_name : str = None : 
	Desired savename, excluding format type which will is defined separately.  Make sure to provide file path if different directory is desired.

	save_format : str = "png" : 
	Save format for plot, .png is default, other standard plt.savefig options available such as "jpg", "pdf", etc.

	continue_plot : bool = False : 
	If desired to neither savefig nor show plot, but plot something else in addition afterwards.  
	Such as calling this plot_funcFit mupltiple times for different data sets to compare.  *Make sure to change colormap/colors accordingly

	font_size : float = 12 :
	font size for text in plot.  Both axes will have this exact float pt size, legend will be 2 pts smaller.

	**func_toplot_kwargs : 
	Any additional kwargs to pass through to func_toPlot, such as constants, extra kwargs, etc.


	Returns
	--------------

	None

	Either shows or saves the plot according to your choice in "save_name : str = None" above.

	'''

	if best_fit_params:
		print(func_toPlot(x_showFit_vals,**best_fit_params,**func_toplot_kwargs).shape)
		if list(x_showFit_vals):
			plt.plot(x_showFit_vals,func_toPlot(x_showFit_vals,**best_fit_params,**func_toplot_kwargs),label="Best Fit",color=cmap(1.0),marker="")
		else:
			x_showFit_vals=np.linspace(np.min(x_values)*0.9,np.max(x_values)*1.1,num=50)
			plt.plot(x_showFit_vals,func_toPlot(x_showFit_vals,**best_fit_params,**func_toplot_kwargs),label="Best Fit",color=cmap(1.0),marker="")
		if fit_param_combos:
			mins=[]
			maxes=[]
			for p in fit_param_combos:
				###Check if any parameters need scaling
				if param_scale:
					for i in param_scale:
						fit_param_combos[p][i]*=param_scale[i]
				if linear_paramNames is not None:
					linears=np.asarray([fit_param_combos[p][l] for l in linear_paramNames])
					nonlin=dict(fit_param_combos[p])
					for l in linear_paramNames:
						del nonlin[l]
					###Check to ensure param_combos isn't being affected
					# print(nonlin.keys())
					# print(param_combos[p].keys())
					constant=func_toPlot(x_showFit_vals[...,None],**nonlin,**dict([[l,0] for l in linear_paramNames]),**func_toplot_kwargs)
					linear_vals=[func_toPlot(x_showFit_vals[...,None],**nonlin,**dict([[l,0] for l in linear_paramNames if l!=i]),**dict([[i,1]]),**func_toplot_kwargs) for i in linear_paramNames]
					result=constant
					for l in range(len(linear_vals)):
						result=result+linears[l]*(linear_vals[l]-constant)
				else:
					result=func_toPlot(x_showFit_vals[...,None],**p,**func_toplot_kwargs)
				mins.append(result.min(axis=1))
				maxes.append(result.max(axis=1))
			for n in range(len(mins))[::-1]:
				###Have to plot in reverse to ensure the narrower bands are not covered up...
				if discrete_colors:
					plt.fill_between(x_showFit_vals,maxes[n],mins[n],label=list(fit_param_combos.keys())[n],color=discrete_colors[n],alpha=0.5)
				else:
					plt.fill_between(x_showFit_vals,maxes[n],mins[n],label=list(fit_param_combos.keys())[n],color=cmap(0.95-0.95*(float(list(fit_param_combos.keys())[n][:-9])-1)/(cmap_sigma_max)),alpha=0.5)		###cmap(0.9-(float(list(fit_param_combos.keys())[n][:-9])-1)/(len(fit_param_combos)))

	###Plotting values, moving to last so they always show above sigma fill-betweens...
	if yerrs:
		plt.errorbar(x_values,y_values,yerr=yerrs,label=val_label,color=cmap(1.0))
	else:
		plt.scatter(x_values,y_values,label=val_label,color=cmap(1.0))

	plt.legend(fontsize=(font_size-2))
	if xlims:
		plt.xlim(xlims)
	if ylims:
		plt.ylim(ylims)
	if xscale:
		plt.xscale(xscale)
	if yscale:
		plt.yscale(yscale)
	
	plt.xlabel(axis_labels[0],size=font_size)
	plt.ylabel(axis_labels[1],size=font_size)
	plt.tick_params(labelsize=font_size)
	if grid:
		plt.grid()
	
	if save_name:
		plt.savefig("%s.%s"%(save_name,save_format),dpi=400, bbox_inches="tight")
		plt.cla()
		plt.clf()
		plt.close("all")
		return None
	elif continue_plot:		###Where multiple sets of data or sigma bounds are desired to be plotted.
		return None
	else:
		plt.show()
		plt.cla()
		plt.clf()
		plt.close("all")
		return None	


def cornerplot_from_margs(marg_param_names,margs,margs_1d_ranges,axis_mins,axis_maxs,marg_param_text_names,legend_names=None,showBnds_2D=[1,2,3],
	cmaps = [mpl.colors.LinearSegmentedColormap.from_list("",["white","lightblue","royalblue","navy"])],cmap_sigma_max=4,alpha=0.8,font_size=12,savename=None,save_format="pdf"):
	
	space=0.2
	marg_len=len(marg_param_names)
	# marg_1d_names=["Marginalized over: "+", ".join(marg_param_names[:i]+marg_param_names[i+1:]) for i in range(marg_len)]
	marg_names=[["Marginalized over: "+", ".join(marg_param_names[n] for n in range(marg_len) if (n!=i and n!=j)) for i in range(marg_len)] for j in range(marg_len)]
	marg_names=np.array(marg_names)
	print(marg_names)
	print(marg_names.shape)
	
	fig,axs=plt.subplots(nrows=marg_len,ncols=marg_len,sharex='col',figsize=(1+2*marg_len,1+2*marg_len))
	###Finding the contour bounds
	if not isinstance(margs,list):	###i.e. only one set to plot
		margs=[margs]
	for m in range(len(margs)):
		fit_param_2d_levels=[]
		for p1 in range(marg_len):
			fit_param_2d_levels.append([])
			for p2 in range(marg_len)[:p1]:
				# print(p1,p2)
				###Determining the levels for 1, 2, and 3 sigma (the correct way now...)
				temp=[]
				print(marg_names[p1,p2])
				for s in showBnds_2D:
					temp.append(prob_level(margs[m][marg_names[p1,p2]],sigma_levels_nD_Gaussian(2,s)))
				# print(temp)
				for t in range(len(temp[1:])):
					if temp[t+1]>=temp[t]:
						temp[t+1]=np.copy(temp[t])/1.000001
				fit_param_2d_levels[p1].append(temp)
		print(fit_param_2d_levels)

		for i in range(marg_len):	###Corner plot rows
			for j in range(marg_len):	###Corner plot cols
				# print(i,j)
				if j>i and m==0:
					fig.delaxes(axs[i,j])
				elif i==j:		###1D marginalized distributions
					axs[i,j].set_yticks([])
					# axs[i,j].tick_params(axis="y",width=0,which="both",pad=200)
					if i!=0:
						axs[i,j].get_yaxis().set_visible(False)
					ref=tuple(np.concatenate((np.arange(i),np.arange(i+1,marg_len))))	###Making axes of chi_sq_probs that I need to sum over (all but i axis)
					# print('ref: ',ref)	###CHECK
					axs[i,j].plot(margs_1d_ranges[i],margs[m][marg_names[i,i]]/np.max(margs[m][marg_names[i,i]]),marker="",linestyle="-")
					axs[i,j].set_ylim([0,1.1])
				elif i>j:
					if j!=0:	###i.e. not leftmost
						axs[i,j].sharey(axs[i,0])
						axs[i,j].tick_params(labelleft=False)	###To not show tick labels on inside plots
					# print([fit_param_2d_levels[i][j][::-1][b-1] for b in showBnds_2D])
					axs[i,j].contourf(*np.meshgrid(margs_1d_ranges[j],margs_1d_ranges[i]),margs[m][marg_names[i][j]].T,levels=[fit_param_2d_levels[i][j][::-1][b] for b in range(len(showBnds_2D))]+[1],colors=[cmaps[m](0.95-0.95*(b-1)/(cmap_sigma_max)) for b in showBnds_2D[::-1]],alpha=alpha)
				axs[i,j].minorticks_on()
	for i in range(marg_len):
		if axis_maxs[i]<0.2:		###Have to do this AFTER the plot is generated in order to grab the scientific notation text
			plt.subplots_adjust(wspace=space,hspace=space)	###General default is 0.2, so increasing slightly to prevent any sci notation overlap
			axs[i,0].ticklabel_format(style="sci",scilimits=(0,0))
			axs[-1,i].ticklabel_format(style="sci",scilimits=(0,0))
			axs[i,0].figure.canvas.draw()
			axs[i,0].yaxis.get_offset_text().set_visible(False)
			axs[-1,i].xaxis.get_offset_text().set_visible(False)
			axs[i,0].set_ylabel(marg_param_text_names[i] + "  [" + axs[i,0].get_yaxis().get_offset_text().get_text() + "]",size=font_size)
			axs[-1,i].set_xlabel(marg_param_text_names[i] + "  [" + axs[-1,i].get_xaxis().get_offset_text().get_text() + "]",size=font_size)
		elif axis_maxs[i]>1000:
			plt.subplots_adjust(wspace=space,hspace=space)	###General default is 0.2, so increasing slightly to prevent any sci notation overlap
			axs[i,0].ticklabel_format(style="sci",scilimits=(0,0))
			axs[-1,i].ticklabel_format(style="sci",scilimits=(0,0))
			axs[i,0].figure.canvas.draw()
			axs[i,0].yaxis.get_offset_text().set_visible(False)
			axs[-1,i].xaxis.get_offset_text().set_visible(False)
			axs[i,0].set_ylabel(marg_param_text_names[i] + "  [" + axs[i,0].get_yaxis().get_offset_text().get_text() + "]",size=font_size)
			axs[-1,i].set_xlabel(marg_param_text_names[i] + "  [" + axs[-1,i].get_xaxis().get_offset_text().get_text() + "]",size=font_size)
		else:
			axs[i,0].set_ylabel(marg_param_text_names[i],size=font_size)
			axs[-1,i].set_xlabel(marg_param_text_names[i],size=font_size)
	###Setting upper left corner
	axs[0,0].set_ylabel(marg_param_text_names[0],size=font_size,labelpad=16)
	###now ensuring left subplots have same y-axes as bottom subplot's x-axes...
	for i in range(1,marg_len):
		axs[-1,i].set_xlim(axis_mins[i],axis_maxs[i])
		axs[i,0].set_yticks(axs[-1,i].get_xticks())
		axs[i,0].set_ylim(axs[-1,i].get_xlim())	###I guess the order matters here?? lim needed after ticks or else ticks will overwrite???? Cool.
	
	###placing the legend above center... or attempting to....
	if legend_names:
		proxy = [plt.Rectangle((0,0),1,1,fc = cmaps[i](1.0),alpha=alpha) for i in range(len(legend_names))]
		fig.legend(proxy, legend_names,loc="center",bbox_to_anchor=axs[0,1].get_position(),fontsize=font_size)

	if savename:
		plt.savefig(savename+"."+save_format,format=save_format,dpi=400,bbox_inches="tight")
	else:
		plt.tight_layout()
		plt.show()
	plt.cla()
	plt.clf()
	plt.close("all")	###To save memory


def cont_bnds_funcFit(func_Fit: Callable, x_cont_vals: list or np.ndarray, fit_param_combos: dict, linear_paramNames: list or np.ndarray = [],
	**func_Fit_kwargs):
	###Check if any parameters need scaling
	if linear_paramNames:	###This does not actually make it faster right now... need to figure out a way to...
		linears=np.asarray([fit_param_combos[l] for l in linear_paramNames])
		print(linears.shape)
		nonlin_kys=list(fit_param_combos.keys())
		# print(nonlin_kys)
		for l in linear_paramNames:
			nonlin_kys.remove(l)
		print("nonlinear keys: ",nonlin_kys)
		nonlinears=np.asarray([fit_param_combos[nl] for nl in nonlin_kys])
		nonlinears_unique,nonlinears_unique_inverse=np.unique(nonlinears,axis=-1,return_inverse=True)
		print("-----------------")
		if nonlinears.size==0 or (len(linear_paramNames)+1)*nonlinears_unique.shape[-1]<nonlinears.shape[-1]:			###i.e. number of combos to pass through are worth it (time-wise...)
			print("linear method employed")
			nonlinears_unique=dict([[nonlin_kys[nl],nonlinears_unique[nl]] for nl in range(len(nonlin_kys))])
			###Constant term when linears==0
			constant=func_Fit(x_cont_vals[...,None],**nonlinears_unique,**dict([[l,0] for l in linear_paramNames]),**func_Fit_kwargs)
			###Now each linear term (when all other linears are still 0), subtracting constant term
			linear_terms=[(func_Fit(x_cont_vals[...,None],**nonlinears_unique,**dict([[l,0] for l in linear_paramNames if l!=i]),**dict([[i,1]]),**func_Fit_kwargs)-constant) for i in linear_paramNames]
			
			###Now add them together to get result for each combo
			if len(nonlin_kys)==0:	###I.e. only 1 combo was passed through each linear term calculation and constant
				result=constant
				for l in range(len(linear_paramNames)):
					result=result+linear_terms[l]*linears[l]
			else:
				constant=constant[...,nonlinears_unique_inverse]
				result=constant
				for l in range(len(linear_paramNames)):
					result=result+linear_terms[l][...,nonlinears_unique_inverse]*linears[l]
		else:
			result=func_Fit(x_cont_vals[...,None],**fit_param_combos,**func_Fit_kwargs)
	else:
		result=func_Fit(x_cont_vals[...,None],**fit_param_combos,**func_Fit_kwargs)
	# print(result.shape)
	mins=result.min(axis=1)
	maxes=result.max(axis=1)
	return mins, maxes

def linearFunc(x,a,b,c):#,d):
	return a+b*x+c*x**2#+d*x**2

def tSZ_Dust220Intensity_2Comp(freq,z,B_dust,T_dust,A_y,A_dust,dust_v_0=353):###dust_v_0==[GHz] #metric=LambdaCDM(H0=68,Om0=0.31,Ode0=0.69,Ob0=0.049,Tcmb0=2.725) ###metric not needed, IDK why this was in it to begin with
	'''Where freq== 2 rows: frequency band [row 0] and weights [row 1].  dust_v_0 is dust reference/rest frequency to scale to'''
	
	###Constants
	T_CMB = 2.725 #K
	k = 1.38*10**(-23) 	#[J/K]
	h = 6.626*10**(-34) #[J*s]
	c = 2.99792458e8	#[m/s]
	x_freq=freq[0]*10**9*h/k/T_CMB
	# xd_freq=freq[0]*10**9*h/k/T_dust
	# xd_220=220*10**9*h/k/T_dust
	band=freq[1]

	def dust_func(v_f):	### v_f in [GHz]
		x_f=v_f*10**9*h/k/T_dust	###Unitless frequency
		###Now scaling by dust_v_0 GHz as reference frequency of gray-body.  Making sure to including constant term in front for safety...
		return 2*(k*T_dust)**3 / (h*c)**2 * (v_f/dust_v_0)**B_dust * (x_f)**3/(np.exp(x_f)-1)
	
	def dBdT_func(x_f):		###In T_CMB units, REMEMBER THE CONSTANT TERM OUT IN FRONT....
		return 2*k*(k*T_CMB)**2 / (h*c)**2 *(x_f)**4*np.exp(x_f)/(np.exp(x_f)-1)**2
	
	# print(band.shape)
	band_sum=spint.trapz(band,x=freq[0],axis=0)	###Sum/Integral of frequency band weights to divide frequency spectrum by (when doing weighted avgs)

	# print(freq.shape,band.shape)
	###Compton-y weighted averages
	y_int=spint.simps(band*T_CMB*(x_freq*(np.exp(x_freq)+1)/(np.exp(x_freq)-1)-4),x=freq[0],axis=0)/band_sum
	# y_int=1
	Dx_int=spint.simps(band*dust_func(freq[0]*(1+z))/dBdT_func(x_freq),x=freq[0],axis=0)/(1+z)/band_sum/dust_func(dust_v_0)			####Now factor of (1+z)/(1+z)^2
	# Dx_int=1
	return A_y*y_int + A_dust*Dx_int


def projAngularAvg_KingOnly_gaussianConvolved(theta_bnds,z,r_0,slope,A_k,beam_fwhm,metric=LambdaCDM(H0=68,Om0=0.31,Ode0=0.69,Ob0=0.049,Tcmb0=2.725),
	dist="comoving",slope_like_powerLaw:bool = True):
	'''Projected angular [arcmin] profile average for a unprojected king k(r)=A_k*(1+(r/r_0)^2)^(-3*slope/2) between bounds of theta_bnds.
	slope_like_powerLaw=True makes the unprojected power of -gamma/2 (matching power law of -gamma), 
	instead of -3*gamma/2 (other standard convention resulting in gamma=1 for projected power equal to 1)'''
	###Setup
	sig=beam_fwhm/np.sqrt(8.*np.log(2))
	if dist.lower()=="comoving":
		D_arcmin=metric.comoving_transverse_distance(z).value*np.pi/180/60
	elif dist.lower()=="proper":
		D_arcmin=metric.angular_diameter_distance(z).value*np.pi/180/60
	else:
		print("dist not recognized... assuming you meant comoving...")
		D_arcmin=metric.transverse_comoving_distance(z).value*np.pi/180/60
	
	if type(theta_bnds)==list:		###i.e. if list of bnds, each of len==2 (or just one list of 2 floats...)
		theta_bnds=np.stack(theta_bnds,axis=-1)
	###Defining inner integral function, amplitude now removing that r_0 term for simplicity. NOTE: i0e has a constant exp(-np.abs(x)) in front, hence why we have np.exp((-(theta-t)**2)/2/sig**2)
	if slope_like_powerLaw:
		kingGauss=lambda  t, theta: theta*A_k*spsp.gamma(1/2)*spsp.gamma((slope-1)/2)/spsp.gamma(slope/2)*(1+(D_arcmin*t/r_0)**2)**((1-slope)/2)*1/sig**(2)*t*np.exp((-(theta-t)**2)/2/sig**2)*spsp.i0e(theta*t/sig**2)
	else:	
		kingGauss=lambda  t, theta: theta*A_k*spsp.gamma(1/2)*spsp.gamma((3*slope-1)/2)/spsp.gamma(3*slope/2)*(1+(D_arcmin*t/r_0)**2)**((1-3*slope)/2)*1/sig**(2)*t*np.exp((-(theta-t)**2)/2/sig**2)*spsp.i0e(theta*t/sig**2)
	###Defining full integral function (double integral using quad_vec to allow for vectorizing of z, r_0, slope, A_k, and beam_fwhm)
	intfunc=lambda b_lower,b_upper: 2*np.pi*spint.quad_vec(lambda theta: spint.quad_vec(lambda x: kingGauss(x, theta), 0, np.inf,workers=1)[0], b_lower, b_upper,workers=1)[0]		###only care about integral return, not error (always is good from my separate testing)
	if type(theta_bnds[0])!=np.ndarray:		###Assume theta_bnds was given as 1D array or list
		return intfunc(theta_bnds[0],theta_bnds[1])/(rad_avg_area(theta_bnds[0],theta_bnds[1]))
	else:	###Should be 2D array with 2 rows (0==lower bnd, 1==upper)  Flattening for any larger arrays needed (*cough* curve fitting *cough*)
		t_lower=theta_bnds[0].flatten()
		t_upper=theta_bnds[1].flatten()
		return np.asarray([intfunc(t_lower[i],t_upper[i])/(rad_avg_area(t_lower[i],t_upper[i])) for i in range(len(t_lower))])		###2 pi factors cancel each other out

def projAngularAvg_PtSrcPlusKing_gaussianConvolved(theta_bnds,z,r_0,slope,A_k,A_ps,beam_fwhm,C=0,metric=LambdaCDM(H0=68,Om0=0.31,Ode0=0.69,Ob0=0.049,Tcmb0=2.725),
	dist="comoving",beam_lmax=None,slope_like_powerLaw:bool = True):
	'''Projected angular [arcmin] profile average for a projected king k(r)=A_k/r_0*(1+(r/r_0)^2)^(-3*slope/2) between bounds of theta_bnds.
	slope_like_powerLaw=True makes the unprojected power of -gamma/2 (matching power law of -gamma), 
	instead of -3*gamma/2 (other standard convention resulting in gamma=1 for projected power equal to 1)'''
	###Setup
	sig=beam_fwhm/np.sqrt(8.*np.log(2))
	arcminToRadian=np.pi/180/60
	if dist.lower()=="comoving":
		D_arcmin=metric.comoving_transverse_distance(z).value*np.pi/180/60
	elif dist.lower()=="proper":
		D_arcmin=metric.angular_diameter_distance(z).value*np.pi/180/60
	else:
		print("dist not recognized... assuming you meant comoving...")
		D_arcmin=metric.comoving_transverse_distance(z).value*np.pi/180/60
	
	if type(theta_bnds)==list:		###i.e. if list of bnds, each of len==2 (or just one list of 2 floats...)
		theta_bnds=np.stack(theta_bnds,axis=-1)
	# print(theta_bnds)
	###Defining inner integral function (extra theta term to be integrated in intfunc) NOTE: i0e has a constant exp(-np.abs(x)) in front, hence why we have np.exp((-(theta-t)**2)/2/sig**2)
	if slope_like_powerLaw:
		kingGauss=lambda  t, theta: theta*A_k*spsp.gamma(1/2)*spsp.gamma((slope-1)/2)/spsp.gamma(slope/2)*(1+(D_arcmin*t/r_0)**2)**((1-slope)/2)*1/sig**(2)*t*np.exp((-(theta-t)**2)/2/sig**2)*spsp.i0e(theta*t/sig**2)
	else:	
		kingGauss=lambda  t, theta: theta*A_k*spsp.gamma(1/2)*spsp.gamma((3*slope-1)/2)/spsp.gamma(3*slope/2)*(1+(D_arcmin*t/r_0)**2)**((1-3*slope)/2)*1/sig**(2)*t*np.exp((-(theta-t)**2)/2/sig**2)*spsp.i0e(theta*t/sig**2)
	###Defining full integral function (double integral using quad_vec to allow for vectorizing of z, r_0, slope, A_k, and beam_fwhm)
	if beam_lmax:
		beam_bl=hp.gauss_beam(beam_fwhm*arcminToRadian,beam_lmax)
		###Okay, new way, as quad doesn't like healpy or visa-versa...
		arc_steps=np.linspace(0,50*sig,num=2000)	###For easier adjusting
		s_norm=2*np.pi*spint.simps(arc_steps*hp.bl2beam(beam_bl,arcminToRadian*arc_steps))*(np.diff(arc_steps[:2]))	###Radial integral to 2d, accounting for discrete int. stepsize
		def intfunc(b_lower,b_upper):
			radbin_steps=np.linspace(b_lower,b_upper,num=int(np.round((b_upper-b_lower)*50)))	###For easier adjusting
			return 2*np.pi*(spint.quad_vec(lambda theta: spint.quad_vec(lambda x: kingGauss(x, theta), 0, np.inf)[0], b_lower, b_upper)[0] + A_ps*spint.simps(radbin_steps*hp.bl2beam(beam_bl,arcminToRadian*radbin_steps))*(np.diff(radbin_steps[:2]))/s_norm)
		###only care about integral return, not error (always is good from my separate testing).  
	else:
		intfunc=lambda b_lower,b_upper: spint.quad_vec(lambda theta: spint.quad_vec(lambda x: kingGauss(x, theta), 0, np.inf)[0], b_lower, b_upper)[0] + A_ps*(np.exp(-(b_lower/sig)**2/2)-np.exp(-(b_upper/sig)**2/2))
	if type(theta_bnds[0])!=np.ndarray:		###Assume theta_bnds was given as 1D array or list
		return intfunc(theta_bnds[0],theta_bnds[1])/(rad_avg_area(theta_bnds[0],theta_bnds[1]))+C
	else:	###Should be 2D array with 2 rows (0==lower bnd, 1==upper)  Flattening for any larger arrays needed (*cough* curve fitting *cough*)
		t_lower=theta_bnds[0].flatten()
		t_upper=theta_bnds[1].flatten()
		return np.asarray([intfunc(t_lower[i],t_upper[i])/(rad_avg_area(t_lower[i],t_upper[i])) for i in range(len(t_lower))])+C

def projAngularAvg_PtSrcOnly_gaussianConvolved(theta_bnds,A_ps,beam_fwhm,C=0,beam_lmax=None):
	'''Projected angular [arcmin] profile average for a central point source. 
	*NOW* With optional additional uniform offset constant C added'''
	###Setup
	sig=beam_fwhm/np.sqrt(8.*np.log(2))
	arcminToRadian=np.pi/180/60
	if type(theta_bnds)==list:		###i.e. if list of bnds, each of len==2 (or just one list of 2 floats...)
		theta_bnds=np.stack(theta_bnds,axis=-1)
	
	if beam_lmax:
		beam_bl=hp.gauss_beam(beam_fwhm*arcminToRadian,beam_lmax)
		###Okay, new way, as quad doesn't like healpy or visa-versa...
		arc_steps=np.linspace(0,50*sig,num=2000)	###For easier adjusting
		s_norm=2*np.pi*spint.simps(arc_steps*hp.bl2beam(beam_bl,arcminToRadian*arc_steps))*(np.diff(arc_steps[:2]))	###Radial integral to 2d, accounting for discrete int. stepsize
		def func(b_lower,b_upper):
			radbin_steps=np.linspace(b_lower,b_upper,num=int(np.round((b_upper-b_lower)*50)))	###For easier adjusting
			return 2*np.pi*(A_ps*spint.simps(radbin_steps*hp.bl2beam(beam_bl,arcminToRadian*radbin_steps))*(np.diff(radbin_steps[:2]))/s_norm)
		###only care about integral return, not error (always is good from my separate testing).  
	else:
		func=lambda b_lower,b_upper: A_ps*(np.exp(-(b_lower/sig)**2/2)-np.exp(-(b_upper/sig)**2/2))
	if type(theta_bnds[0])!=np.ndarray:		###Assume theta_bnds was given as 1D array or list
		return func(theta_bnds[0],theta_bnds[1])/(rad_avg_area(theta_bnds[0],theta_bnds[1]))+C
	else:	###Should be 2D array with 2 rows (0==lower bnd, 1==upper)  Flattening for any larger arrays needed (*cough* curve fitting *cough*)
		t_lower=theta_bnds[0].flatten()
		t_upper=theta_bnds[1].flatten()
		return np.asarray([func(t_lower[i],t_upper[i])/(rad_avg_area(t_lower[i],t_upper[i])) for i in range(len(t_lower))])+C

def PtSrcOnly_gaussianConvolved(thetas,A_ps,beam_fwhm,beam_lmax=None):
	sig=beam_fwhm/np.sqrt(8.*np.log(2))
	thetas=np.array(thetas)
	thetas_shape=thetas.shape
	thetas=np.ravel(thetas)
	if beam_lmax:
		arcminToRadian=np.pi/180/60
		beam_bl=hp.gauss_beam(beam_fwhm*arcminToRadian,beam_lmax)
		theta_steps=np.linspace(0,25*sig,num=2000)
		s_norm=2*np.pi*np.diff(theta_steps[:2])*spint.simps(theta_steps*hp.bl2beam(beam_bl,arcminToRadian*theta_steps))
		return A_ps*(hp.bl2beam(beam_bl,thetas*arcminToRadian)/s_norm).reshape(thetas_shape)
	else:
		return A_ps/sig**2*np.exp(-(thetas.reshape(thetas_shape)/sig)**2/2)

def KingOnly_gaussianConvolved(thetas,z,r_0,slope,A_k,beam_fwhm,metric=LambdaCDM(H0=68,Om0=0.31,Ode0=0.69,Ob0=0.049,Tcmb0=2.725),dist="comoving",slope_like_powerLaw:bool = True):
	'''slope_like_powerLaw=True makes the unprojected power of -gamma/2 (matching power law of -gamma), 
	instead of -3*gamma/2 (other standard convention resulting in gamma=1 for projected power equal to 1)'''
	sig=beam_fwhm/np.sqrt(8.*np.log(2))
	if dist.lower()=="comoving":
		D_arcmin=metric.comoving_transverse_distance(z).value*np.pi/180/60
	elif dist.lower()=="proper":
		D_arcmin=metric.angular_diameter_distance(z).value*np.pi/180/60
	else:
		print("dist not recognized... assuming you meant comoving...")
		D_arcmin=metric.transverse_comoving_distance(z).value*np.pi/180/60
	if slope_like_powerLaw:	###NOTE: i0e has a constant exp(-np.abs(x)) in front, hence why we have np.exp((-(theta-t)**2)/2/sig**2)
		kingGauss=lambda  t: spsp.gamma(1/2)*spsp.gamma((slope-1)/2)/spsp.gamma(slope/2)*(1+(D_arcmin*t/r_0)**2)**((1-slope)/2)*1/sig**(2)*t*np.exp((-(thetas-t)**2)/2/sig**2)*spsp.i0e(thetas*t/sig**2)
	else:
		kingGauss=lambda  t: spsp.gamma(1/2)*spsp.gamma((3*slope-1)/2)/spsp.gamma(3*slope/2)*(1+(D_arcmin*t/r_0)**2)**((1-3*slope)/2)*1/sig**(2)*t*np.exp((-(thetas-t)**2)/2/sig**2)*spsp.i0e(thetas*t/sig**2)
	return A_k*np.asarray(spint.quad_vec(kingGauss, 0, np.inf,workers=1)[0])

def PtSrcPlusKing_gaussian_Convolved(thetas,z,r_0,slope,A_k,A_ps,beam_fwhm,C=0,metric=LambdaCDM(H0=68,Om0=0.31,Ode0=0.69,Ob0=0.049,Tcmb0=2.725),dist="comoving",beam_lmax=None,slope_like_powerLaw:bool = True):
	return PtSrcOnly_gaussianConvolved(thetas,A_ps,beam_fwhm,beam_lmax)+KingOnly_gaussianConvolved(thetas,z,r_0,slope,A_k,beam_fwhm,metric=metric,dist=dist,slope_like_powerLaw=slope_like_powerLaw)+C

def proj_king_only(rs,r_0,slope,A_k,slope_like_powerLaw:bool = True):
	'''slope_like_powerLaw=True makes the unprojected power of -gamma/2 (matching power law of -gamma), 
	instead of -3*gamma/2 (other standard convention resulting in gamma=1 for projected power equal to 1)'''
	if slope_like_powerLaw:
		return A_k*spsp.gamma(1/2)*spsp.gamma((slope-1)/2)/spsp.gamma(slope/2)*(1+(rs/r_0)**2)**((1-slope)/2)
	else:
		return A_k*spsp.gamma(1/2)*spsp.gamma((3*slope-1)/2)/spsp.gamma(3*slope/2)*(1+(rs/r_0)**2)**((1-3*slope)/2)

def proj_powerlaw_only(rs,r_0,slope,A_pl):
	return A_pl*spsp.gamma(1/2)*spsp.gamma((slope-1)/2)/spsp.gamma(slope/2)*(rs/r_0)**(1-slope)

###Some smaller stuff for mean subtracting and covariances scaling
def rad_avg_area(r0,r1):
	return (r1**2-r0**2)*np.pi

def sub_outerN(array,rads,N=3,returnMeanShift=False):
	'''To normalize offset of furthest N bins (radial), returnMeanShift returns value subtracted from each element'''
	# if len(array.shape)==1:
	# 	mshift=np.sum(array[-N:]*rad_avg_area(rads[-(N+1):-1],rads[-N:]))/rad_avg_area(rads[-(N+1)],rads[-1])
	# elif len(array.shape)==2:
	# 	mshift=np.sum(array[:,-N:]*rad_avg_area(rads[-(N+1):-1],rads[-N:]))/rad_avg_area(rads[-(N+1)],rads[-1])
	# else:
	# print(rad_avg_area(rads[-(N+1):-1],rads[-N:]))
	# print(rad_avg_area(rads[-(N+1)],rads[-1]))
	mshift=np.sum(np.array(array)[...,-N:]*rad_avg_area(rads[-(N+1):-1],rads[-N:]),axis=-1)/rad_avg_area(rads[-(N+1)],rads[-1])
	# print(mshift)
	if returnMeanShift:
		return np.array(array)-mshift[...,None], mshift
	else:
		return np.array(array)-mshift[...,None]

def radavgs_toSum_within(df,rads,rad_max,N=3,average_rows=True):
	col_max=np.max(np.where(rads<=rad_max))
	###condense to radavg profile first: (if a request catalog, optional if already done)
	if average_rows:
		df=df.mean(axis=0)
	###Mean correcting edge and then only taking the bins within desired sum:
	meaned=sub_outerN(df,rads=rads,N=N)[...,:col_max]
	areas=rad_avg_area(rads[:col_max],rads[1:col_max+1])
	return np.sum(meaned*areas,axis=-1)		###returns sum of means*areas (cumulative sum to given radius)

def scale_covs(covs,N=10):
	'''Where covs is bxfxf array of b radial bins, N= # of bins at end that are re-scaled'''
	if covs.shape[-2]!=covs.shape[-1]:
		print("Array doesn't look like bxfxf set of b cov matrices")
		return None
	fs=covs.shape[-1]
	for f1 in range(fs):
		for f2 in range(fs):
			covs[-N:,f1,f2]=covs[-N:,f1,f2]*np.sqrt(covs[-(N+1),f1,f1])*np.sqrt(covs[-(N+1),f2,f2])/np.sqrt(covs[-N:,f1,f1])/np.sqrt(covs[-N:,f2,f2])
	return covs

def x_y_pl(x,x_peak,pl_amplitude,pl_slope):
	return pl_amplitude*(x/x_peak)**pl_slope

def xlog_y_pl(xlog,xlog_peak,pl_amplitude,pl_slope):
	return np.log10(pl_amplitude)+(xlog-xlog_peak)*pl_slope

def xlog_ylog_pl(xlog,xlog_peak,log_pl_amplitude,pl_slope):
	return log_pl_amplitude+(xlog-xlog_peak)*pl_slope

def energy_mass_correction_by_convolving(m_range: list or np.ndarray, pl_amplitude: float, pl_slope: float, sed_sigma: float, 
mass_distr_sigmas: float or list or np.ndarray, mass_distr_means: float or list or np.ndarray,E_lin: bool = True):
	'''Quick way when mass_distr_means= single mean (i.e. mass distr. is Gaussian)'''
	
	def real_sigma(old_sigma,factored_sigma):
		return np.sqrt(old_sigma**2-factored_sigma**2)
	
	def Gaussian(x,amp,c,sig):
		return amp/(sig*np.sqrt(2*np.pi)) * np.exp(-((x-c)**2)/2/sig**2)
	
	uncon_sigma=real_sigma(mass_distr_sigmas,sed_sigma)
	if E_lin:
		def E_w(mass):
			return xlog_y_pl(mass,xlog_peak=mass_distr_means,pl_amplitude=(pl_amplitude),pl_slope=pl_slope)*Gaussian(mass,1,mass_distr_means,uncon_sigma)
	else:
		def E_w(mass):
			return xlog_ylog_pl(mass,xlog_peak=mass_distr_means,log_pl_amplitude=pl_amplitude,pl_slope=pl_slope)*Gaussian(mass,1,mass_distr_means,uncon_sigma)

	# plt.plot(m_range,Gaussian(m_range,1,mass_distr_means,uncon_sigma),label="w_G",marker="")
	# plt.plot(m_range,spi.gaussian_filter1d(E_w(m_range),sed_sigma),label="E_w_convolved",marker="")
	# plt.plot(m_range,E_w(m_range),label="E_w",marker="")
	# plt.plot(m_range,energy_mass_log_pl(m_range,logm_peak=mass_distr_means,pl_amplitude=(pl_amplitude),pl_slope=pl_slope),label="E",marker="")
	# # plt.plot(m_range,E_w(m_range)/Gaussian(m_range,1,mass_distr_means,uncon_sigma),label="E_w/w_G",marker="")
	# plt.legend()
	# plt.yscale("log")
	# plt.show()
	# plt.close()
	# exit()
	# return np.convolve(E_w(m_range),Gaussian(m_range,1,np.mean(m_range),sed_sigma),mode="same")/np.convolve(Gaussian(m_range,1,mass_distr_means,uncon_sigma),Gaussian(m_range,1,np.mean(m_range),sed_sigma),mode="same")
	print(E_w(m_range).shape,Gaussian(m_range,1,np.mean(m_range),sed_sigma).shape,Gaussian(m_range,1,mass_distr_means,uncon_sigma).shape)
	if len(m_range.shape)>1:	###Checking if multidimensional and then second axis is longer than length of 1 (which will be raveled...)
		if m_range.shape[1]>1:
			print("Cannot pass multidimensional m_range through this function due to convolution... ending before an error or incorrect value is returned",time.asctime())
			exit()
		g1=Gaussian(m_range.ravel(),1,np.mean(m_range),sed_sigma)
		g2=Gaussian(m_range.ravel(),1,mass_distr_means,uncon_sigma)
		return spi.convolve1d(E_w(m_range),g1,mode="constant",origin=0,axis=0)/spi.convolve1d(g1,g2,mode="constant",origin=0)[...,None]
	else:	###i.e. m_range is a 1d array, which is fine
		g1=Gaussian(m_range,1,np.mean(m_range),sed_sigma)
		g2=Gaussian(m_range,1,mass_distr_means,uncon_sigma)
		return spi.convolve1d(E_w(m_range),g1,mode="constant",origin=0,axis=-1)/spi.convolve1d(g1,g2,mode="constant",origin=0)
	# return spi.gaussian_filter1d(E_w(m_range),sed_sigma/np.diff(m_range[:2]),mode="nearest")/spi.gaussian_filter1d((Gaussian(m_range,1,mass_distr_means,uncon_sigma)),sed_sigma/np.diff(m_range[:2]),mode="nearest")

def energy_mass_correction(mbin_edges: list or np.ndarray, pl_amplitude: float, pl_slope: float, sed_sigma: float, 
mass_distr_sigmas: float or list or np.ndarray, mass_distr_means: float or list or np.ndarray, norms: list or np.ndarray=None, E_lin: bool=True):
	'''Mass correction of a given set of mass bin measurements (given in log10(M/M_sun) dex), 
	where the mass distribution is described by a set of Gaussians and said masses have their own sed_sigma and 
	the expected unconvolved function is a power-law: A*(M/M_p)^alpha.  M_p is the peak mass as determined by the provided gaussian fit of the original mass distribution.
	**Notes: sed_sigma must be >0.08 else may cause a RuntimeWarning with scipy.integrate.quad_vec and scipy.special.erfc()

	
	Parameters
	--------------

	mbin_edges: list or np.ndarray
	1d mass bin edges of bin values.  Must be at least length 2.

	pl_amplitude: float
	Power law Amplitude

	pl_slope: float
	Power law Slope
	
	sed_sigma: float
	Uncertainty in the mass from SED fitting that is being factored out of mass bin.
	
	mass_distr_sigmas: float or list or np.ndarray
	sigmas of the mass distribution.  Float = single gaussian fit sigma.  
	list or np.ndarray = multiple gaussian fit sigmas.
	
	mass_distr_means: float or list or np.ndarray
	means of the mass distribution.  Float = single gaussian fit mean.  
	list or np.ndarray = multiple gaussian fit means.
	
	norms: list or np.ndarray = None
	Amplitude norms per Gaussian when using multiple Gaussians.  Otherwise will not be used.  
	Only relative amplitude norms are needed, will be fully normalized to zero.

	E_lin: bool = True
	If E_peak is sought in it's linear form (i.e. for fitting, equal steps in linear), else False = log form instead
	
	'''

	def real_sigma(old_sigma,factored_sigma):
		return np.sqrt(old_sigma**2-factored_sigma**2)
	
	def Gaussian(x,amp,c,sig):
		return amp/(sig*np.sqrt(2*np.pi)) * np.exp(-((x-c)**2)/2/sig**2)

	def Gint(mass,sigma,lower,upper):
		return (spsp.erf((upper-mass)/sigma/np.sqrt(2))-spsp.erf((lower-mass)/sigma/np.sqrt(2)))/2

	def Gint2(mass,sigma,lower,upper):	###New, slightly better way to not cause the weird runTime error quite as fast...
		return spint.quad_vec(lambda x: Gaussian(x,1,mass,sigma),lower,upper)[0]

	# print(Gint2(mass_distr_means,sed_sigma,1e-10,1e15))
	# print(spint.quad(lambda x: Gaussian(np.log10(x),1,mass_distr_means,sed_sigma),1e-10,1e14,epsrel=-1,points=[mass_distr_means]))
	# x=10**np.arange(10.5,12.5,0.01)
	# plt.plot(x,Gaussian(np.log10(x),1,mass_distr_means,sed_sigma))
	# # plt.xscale("log")
	# # plt.show()
	# plt.close()
	# exit()
	# print(mass_distr_sigmas)
	# print(mass_distr_means)
	if norms is not None:
		print("Mass distribution not an individual Gaussian...")
		norms=np.array(norms)/np.sum(norms)			###Normalizing the norms
		uncon_sigmas=real_sigma(np.array(mass_distr_sigmas),sed_sigma)
		m_check=np.arange(0,20*np.max(mbin_edges),0.01)
		mass_peak=10**m_check[np.argmax(np.sum(Gaussian(m_check[...,None],norms,np.array(mass_distr_means),np.array(mass_distr_sigmas)),axis=1))]
		logmass_peak=m_check[np.argmax(np.sum(Gaussian(m_check[...,None],norms,np.array(mass_distr_means),np.array(mass_distr_sigmas)),axis=1))]	###For log method (to validate that I get the same answer as old way)
		def E_weight(mass,m_lower,m_upper):
			return np.sum(Gaussian(mass,norms,np.array(mass_distr_means),np.array(uncon_sigmas)))*Gint2(mass,sed_sigma,m_lower,m_upper)
	else:
		uncon_sigma=real_sigma(mass_distr_sigmas,sed_sigma)
		mass_peak=10**mass_distr_means
		logmass_peak=mass_distr_means	###For log method (to validate that I get the same answer as old way)
		def E_weight(mass,m_lower,m_upper):
			return Gaussian(mass,1,mass_distr_means,uncon_sigma)*Gint2(mass,sed_sigma,m_lower,m_upper)
			# return Gint2(mass,sed_sigma,m_lower,m_upper)	###Equal weighting
	if E_lin:
		def E(mass,m_lower,m_upper):
			###Log-log method (bc fitting procedure if given is_linear=True plugs in zero...)
			return (xlog_y_pl(mass,xlog_peak=logmass_peak,pl_amplitude=(pl_amplitude),pl_slope=pl_slope))*E_weight(mass,m_lower,m_upper)	
	else:
		def E(mass,m_lower,m_upper):
			return xlog_ylog_pl(mass,xlog_peak=logmass_peak,log_pl_amplitude=pl_amplitude,pl_slope=pl_slope)*E_weight(mass,m_lower,m_upper)	
	integral_min=1
	###Setting constant since doesn't matter too much... and we don't expect super massive galaxies...
	integral_max=20 #np.ceil(2*np.max(mass_distr_means)+5*(np.max(mass_distr_sigmas)+sed_sigma))
	print(integral_max)

	###Previous method(s), changed to integrating in log steps and log-log space
	###This was my first "new" attempt, numerical integration in real (non-log) space is painfully inaccurate, but I had forgotten the correction term
	# method1=np.array([spint.quad_vec(lambda x: E(x,mbin_edges[m],mbin_edges[m+1]),0,integral_max)[0]/spint.quad_vec(lambda x: E_weight(x,mbin_edges[m],mbin_edges[m+1]),0,integral_max)[0] for m in range(len(mbin_edges)-1)])
	# if linear:
	# 	linear_check=np.array([spint.quad_vec(lambda x: np.log(10)*10**x*E(x,mbin_edges[m],mbin_edges[m+1],flat=True),mbin_edges[m],mbin_edges[m+1])[0]/spint.quad_vec(lambda x: np.log(10)*10**x*E_weight(x,mbin_edges[m],mbin_edges[m+1],flat=True),mbin_edges[m],mbin_edges[m+1])[0] for m in range(len(mbin_edges)-1)])
	# 	return linear_check
	# else:
	# 	###Method 2 should the correct way to numerically integrate in log space but with linear steps (has the additional np.log(10)*10**x term in both integrals).
	# 	method2=np.array([spint.quad_vec(lambda x: np.log(10)*10**x*E(x,mbin_edges[m],mbin_edges[m+1]),integral_min,integral_max)[0]/spint.quad_vec(lambda x: np.log(10)*10**x*E_weight(x,mbin_edges[m],mbin_edges[m+1]),integral_min,integral_max)[0] for m in range(len(mbin_edges)-1)])
	# 	# oldmethod=10**np.array([spint.quad_vec(lambda x: E(x,mbin_edges[m],mbin_edges[m+1]),integral_min,integral_max)[0]/spint.quad_vec(lambda x: E_weight(x,mbin_edges[m],mbin_edges[m+1]),integral_min,integral_max)[0] for m in range(len(mbin_edges)-1)])
	# 	# print(method1/method2)
	# 	return method2 #oldmethod #method1, method2
	result=10**np.array([spint.quad_vec(lambda x: E(x,mbin_edges[m],mbin_edges[m+1]),integral_min,integral_max)[0]/spint.quad_vec(lambda x: E_weight(x,mbin_edges[m],mbin_edges[m+1]),integral_min,integral_max)[0] for m in range(len(mbin_edges)-1)])
	return result

def energy_mass_correction_mcenters(mbin_centers: list or np.ndarray, mbin_widths: float or list or np.ndarray, pl_amplitude: float, pl_slope: float, sed_sigma: float, 
mass_distr_sigmas: float or list or np.ndarray, mass_distr_means: float or list or np.ndarray, norms: list or np.ndarray=None, E_lin: bool=True):
	'''Mass correction of a given set of mass bin measurements (given in log10(M/M_sun) dex), 
	where the mass distribution is described by a set of Gaussians and said masses have their own sed_sigma and 
	the expected unconvolved function is a power-law: A*(M/M_p)^alpha.  M_p is the peak mass as determined by the provided gaussian fit of the original mass distribution.
	**Notes: sed_sigma must be >0.08 else may cause a RuntimeWarning with scipy.integrate.quad_vec and scipy.special.erfc()

	
	Parameters
	--------------

	mbin_centers: list or np.ndarray
	1d mass bin centers of bin values.

	mbin_widths: float or list or np.ndarray
	Float for equal widths, or 1d list/np.ndarray same size as mbin_centers

	pl_amplitude: float
	Power law Amplitude

	pl_slope: float
	Power law Slope
	
	sed_sigma: float
	Uncertainty in the mass from SED fitting that is being factored out of mass bin.
	
	mass_distr_sigmas: float or list or np.ndarray
	sigmas of the mass distribution.  Float = single gaussian fit sigma.  
	list or np.ndarray = multiple gaussian fit sigmas.
	
	mass_distr_means: float or list or np.ndarray
	means of the mass distribution.  Float = single gaussian fit mean.  
	list or np.ndarray = multiple gaussian fit means.
	
	norms: list or np.ndarray = None
	Amplitude norms per Gaussian when using multiple Gaussians.  Otherwise will not be used.  
	Only relative amplitude norms are needed, will be fully normalized to zero.

	E_lin: bool = True
	If E_peak is sought in it's linear form (i.e. for fitting, equal steps in linear), else False = log form instead
	
	'''

	def real_sigma(old_sigma,factored_sigma):
		return np.sqrt(old_sigma**2-factored_sigma**2)
	
	def Gaussian(x,amp,c,sig):
		return amp/(sig*np.sqrt(2*np.pi)) * np.exp(-((x-c)**2)/2/sig**2)

	def Gint(mass,sigma,lower,upper):
		return (spsp.erf((upper-mass)/sigma/np.sqrt(2))-spsp.erf((lower-mass)/sigma/np.sqrt(2)))/2

	def Gint2(mass,sigma,lower,upper):	###New, slightly better way to not cause the weird runTime error quite as fast...
		return spint.quad_vec(lambda x: Gaussian(x,1,mass,sigma),lower,upper)[0]

	# print(Gint2(mass_distr_means,sed_sigma,1e-10,1e15))
	# print(spint.quad(lambda x: Gaussian(np.log10(x),1,mass_distr_means,sed_sigma),1e-10,1e14,epsrel=-1,points=[mass_distr_means]))
	# x=10**np.arange(10.5,12.5,0.01)
	# plt.plot(x,Gaussian(np.log10(x),1,mass_distr_means,sed_sigma))
	# # plt.xscale("log")
	# # plt.show()
	# plt.close()
	# exit()
	# print(mass_distr_sigmas)
	# print(mass_distr_means)
	if norms is not None:
		print("Mass distribution not an individual Gaussian...")
		norms=np.array(norms)/np.sum(norms)			###Normalizing the norms
		uncon_sigmas=real_sigma(np.array(mass_distr_sigmas),sed_sigma)
		m_check=np.arange(0,20*(np.max(mbin_centers)+np.max(mbin_widths)),0.01)
		mass_peak=10**m_check[np.argmax(np.sum(Gaussian(m_check[...,None],norms,np.array(mass_distr_means),np.array(mass_distr_sigmas)),axis=1))]
		logmass_peak=m_check[np.argmax(np.sum(Gaussian(m_check[...,None],norms,np.array(mass_distr_means),np.array(mass_distr_sigmas)),axis=1))]	###For log method (to validate that I get the same answer as old way)
		def E_weight(mass,m_lower,m_upper):
			return np.sum(Gaussian(mass,norms,np.array(mass_distr_means),np.array(uncon_sigmas)))*Gint2(mass,sed_sigma,m_lower,m_upper)
	else:
		uncon_sigma=real_sigma(mass_distr_sigmas,sed_sigma)
		mass_peak=10**mass_distr_means
		logmass_peak=mass_distr_means	###For log method (to validate that I get the same answer as old way)
		def E_weight(mass,m_lower,m_upper):
			return Gaussian(mass,1,mass_distr_means,uncon_sigma)*Gint2(mass,sed_sigma,m_lower,m_upper)
			# return Gint2(mass,sed_sigma,m_lower,m_upper)	###Equal weighting
	if E_lin:
		def E(mass,m_lower,m_upper):
			###Log-log method (bc fitting procedure if given is_linear=True plugs in zero...)
			return ((xlog_y_pl(mass,xlog_peak=logmass_peak,pl_amplitude=(pl_amplitude),pl_slope=pl_slope)))*E_weight(mass,m_lower,m_upper)	
	else:
		def E(mass,m_lower,m_upper):
			return xlog_ylog_pl(mass,xlog_peak=logmass_peak,log_pl_amplitude=pl_amplitude,pl_slope=pl_slope)*E_weight(mass,m_lower,m_upper)	
	integral_min=1
	###Setting constant since doesn't matter too much... and we don't expect super massive galaxies...
	integral_max=20 #np.ceil(2*np.max(mass_distr_means)+5*(np.max(mass_distr_sigmas)+sed_sigma))
	print(integral_max)

	if isinstance(mbin_widths,(list,np.ndarray)) and len(mbin_widths)==len(mbin_centers): ###if a width is given per bin
		result=10**np.array([spint.quad_vec(lambda x: E(x,mbin_centers[m]-mbin_widths[m]/2,mbin_centers[m]+mbin_widths[m]/2),integral_min,integral_max)[0]/spint.quad_vec(lambda x: E_weight(x,mbin_centers[m]-mbin_widths[m]/2,mbin_centers[m]+mbin_widths[m]/2),integral_min,integral_max)[0] for m in range(len(mbin_centers))])
	elif isinstance(mbin_widths,(int,float)):
		result=10**np.array([spint.quad_vec(lambda x: E(x,mbin_centers[m]-mbin_widths/2,mbin_centers[m]+mbin_widths/2),integral_min,integral_max)[0]/spint.quad_vec(lambda x: E_weight(x,mbin_centers[m]-mbin_widths/2,mbin_centers[m]+mbin_widths/2),integral_min,integral_max)[0] for m in range(len(mbin_centers))])
	return result

def energy_mass_inf_correction(m_range: list or np.ndarray, pl_amplitude: float, pl_slope: float, sed_sigma: float, 
mass_distr_sigmas: float or list or np.ndarray, mass_distr_means: float or list or np.ndarray, norms: list or np.ndarray=None, E_lin: bool=True):
	'''Mass correction of a given set of mass bin measurements (given in log10(M/M_sun) dex), 
	where the mass distribution is described by a set of Gaussians and said masses have their own sed_sigma and 
	the expected unconvolved function is a power-law: A*(M/M_p)^alpha.  M_p is the peak mass as determined by the provided gaussian fit of the original mass distribution.
	**Notes: sed_sigma must be >0.08 else may cause a RuntimeWarning with scipy.integrate.quad_vec and scipy.special.erfc()

	
	Parameters
	--------------

	mbin_edges: list or np.ndarray
	1d mass bin edges of bin values.  Must be at least length 2.

	pl_amplitude: float
	Power law Amplitude

	pl_slope: float
	Power law Slope
	
	sed_sigma: float
	Uncertainty in the mass from SED fitting that is being factored out of mass bin.
	
	mass_distr_sigmas: float or list or np.ndarray
	sigmas of the mass distribution.  Float = single gaussian fit sigma.  
	list or np.ndarray = multiple gaussian fit sigmas.
	
	mass_distr_means: float or list or np.ndarray
	means of the mass distribution.  Float = single gaussian fit mean.  
	list or np.ndarray = multiple gaussian fit means.
	
	norms: list or np.ndarray = None
	Amplitude norms per Gaussian when using multiple Gaussians.  Otherwise will not be used.  
	Only relative amplitude norms are needed, will be fully normalized to zero.

	E_lin: bool = True
	If E_peak is sought in it's linear form (i.e. for fitting, equal steps in linear), else False = log form instead
	
	'''

	def real_sigma(old_sigma,factored_sigma):
		return np.sqrt(old_sigma**2-factored_sigma**2)
	
	def Gaussian(x,amp,c,sig):
		return amp/(sig*np.sqrt(2*np.pi)) * np.exp(-((x-c)**2)/2/sig**2)

	# print(Gint2(mass_distr_means,sed_sigma,1e-10,1e15))
	# print(spint.quad(lambda x: Gaussian(np.log10(x),1,mass_distr_means,sed_sigma),1e-10,1e14,epsrel=-1,points=[mass_distr_means]))
	# x=10**np.arange(10.5,12.5,0.01)
	# plt.plot(x,Gaussian(np.log10(x),1,mass_distr_means,sed_sigma))
	# # plt.xscale("log")
	# # plt.show()
	# plt.close()
	# exit()
	# print(mass_distr_sigmas)
	# print(mass_distr_means)
	if norms is not None:
		print("Mass distribution not an individual Gaussian...")
		norms=np.array(norms)/np.sum(norms)			###Normalizing the norms
		uncon_sigmas=real_sigma(np.array(mass_distr_sigmas),sed_sigma)
		m_check=np.arange(0,20*np.max(m_range),0.01)
		mass_peak=10**m_check[np.argmax(np.sum(Gaussian(m_check[...,None],norms,np.array(mass_distr_means),np.array(mass_distr_sigmas)),axis=1))]
		logmass_peak=m_check[np.argmax(np.sum(Gaussian(m_check[...,None],norms,np.array(mass_distr_means),np.array(mass_distr_sigmas)),axis=1))]	###For log method (to validate that I get the same answer as old way)
		def E_weight(mass,mass_bin):
			return np.sum(Gaussian(mass,norms,np.array(mass_distr_means),np.array(uncon_sigmas)))*Gaussian(mass_bin,1,mass,sed_sigma)#*Gint2(mass,sed_sigma,m_lower,m_upper)
	else:
		uncon_sigma=real_sigma(mass_distr_sigmas,sed_sigma)
		mass_peak=10**mass_distr_means
		logmass_peak=mass_distr_means	###For log method (to validate that I get the same answer as old way)
		def E_weight(mass,mass_bin):
			return Gaussian(mass,1,mass_distr_means,uncon_sigma)*Gaussian(mass_bin,1,mass,sed_sigma)#Gint2(mass,sed_sigma,m_lower,m_upper)
			# return Gint2(mass,sed_sigma,m_lower,m_upper)	###Equal weighting
	if E_lin:
		def E(mass,mass_bin):
			###Log-log method (bc fitting procedure if given is_linear=True plugs in zero...)
			return ((xlog_y_pl(mass,xlog_peak=logmass_peak,pl_amplitude=(pl_amplitude),pl_slope=pl_slope)))*E_weight(mass,mass_bin)	
	else:
		def E(mass,mass_bin):
			return xlog_ylog_pl(mass,xlog_peak=logmass_peak,log_pl_amplitude=pl_amplitude,pl_slope=pl_slope)*E_weight(mass,mass_bin)	
	integral_min=1
	###Setting constant since doesn't matter too much... and we don't expect super massive galaxies...
	integral_max=20 #np.ceil(2*np.max(mass_distr_means)+5*(np.max(mass_distr_sigmas)+sed_sigma))
	# print(integral_max)
	result=10**np.array([spint.quad_vec(lambda x: E(x,m_range[m]),integral_min,integral_max)[0]/spint.quad_vec(lambda x: E_weight(x,m_range[m]),integral_min,integral_max)[0] for m in range(len(m_range))])
	return result

# print(energy_mass_correction(np.arange(10.6,12.01,0.1)-0.05,1,4,sed_sigma=0.16,mass_distr_sigmas=0.2,mass_distr_means=11.2))
# exit()
###Low pass size noise tests
if __name__=="__main__":
	print('Start Time: ', time.asctime() )

	# low_TH=np.arange(1,8.61,0.4)
	# low_G=np.arange(1,8.61,0.4)
	low_TH=[2,2.5,3]
	low_G=[4,5,6]
	print(low_TH)
	print(low_G)
	high=5
	pix_res=0.05
	NSIDE=8192
	arc=60
	c=2.5
	cs=[2,2.5,3]
	gs=[5,5.5,6]
	beam=2.10
	Freq=[95,150,220]
	carc=[2.5,5]
	f2s=1/(2*np.sqrt(np.log(4)))
	# fset=[]
	data=[]
	params='_60.00Arcmin_0.05pixRes.txt'
	iloc='../Stacks/Images/Tests/'
	basic_x=np.arange(6)
	basic_a=np.arange(3)+10
	basic_b=np.arange(2)+5
	basic_c=np.arange(1)
	print(basic_x,basic_a,basic_b,basic_c)
	a_combo,b_combo,c_combo=np.meshgrid(basic_a,basic_b,basic_c,indexing="ij")
	print(a_combo.shape,b_combo.shape,c_combo.shape)
	abc_dict={"a":a_combo.ravel(),"b":b_combo.ravel(),"c":c_combo.ravel()}
	print(abc_dict)
	print(cont_bnds_funcFit(linearFunc,basic_x,fit_param_combos=abc_dict,linear_paramNames=["c","a","b"]))
	print(linearFunc(basic_x,np.min(basic_a),np.min(basic_b),np.min(basic_c)))
	print(linearFunc(basic_x,np.max(basic_a),np.max(basic_b),np.max(basic_c)))
	# exit()
	test=func_param("test",r"test$\mu$",5)
	print(test.func_name,test.text_name,test.fit_range,test.prior_weight,test.constant,test.has_prior)

	test_size=20
	xrand=np.random.random(size=test_size)*10	###Example x_values ranging from 0 to 10
	xnoise=np.random.normal(0.,0.1,size=test_size)
	print("xrand: ",xrand)
	print("xnoise: ",xnoise)
	a,b,c,d=5,2,0.2,0.2
	yrand=linearFunc(xrand+xnoise,a,b,c)#,d)
	# yrand*=np.random.normal(loc=1,scale=0.02,size=test_size)	###To give rand shifting (multiplicatively)
	# yrand+=np.random.normal(loc=0,scale=1,size=test_size)	###To give rand shifting (addition)

	example_classes=[func_param("a",r"$\alpha$",fit_range=a,is_linear=True),func_param("b",r"$\beta$",fit_range=np.arange(1,3.001,0.01),is_linear=True),func_param("c",r"$\gamma$",fit_range=np.arange(0,1.001,0.005))]#,func_param("d",r"$\delta$",fit_range=np.arange(0,0.501,0.005))]
	print(example_classes[0].text_name)
	###Dictionary test example
	dicttest=dict([[ec.func_name,ec.constant] for ec in example_classes])	#([["one",1]])
	print(dicttest)
	
	GLS_fit1,GLS_cov1=funcFit(linearFunc,xrand,yrand,example_classes,variance=np.ones(len(xrand)),method="GLS")
	print(GLS_fit1)
	print(GLS_cov1)
	# GLS_fit2,GLS_cov2=funcFit(linearFunc,xrand,yrand,example_classes,variance=None,method="GLS")
	# print(GLS_fit2)
	# print(GLS_cov2)

	# t=time.time()
	# t_trials=20
	# for i in range(t_trials):
	BE_fit1,BE_cov1,BE_errs1,BE_bnds1=funcFit(linearFunc,xrand,yrand,example_classes,variance=None,method="BE",cornerplot=True,n_d_bnds=[2])[:-1]
	print(BE_fit1)
	print(BE_cov1)
	print(BE_errs1)
	print(BE_bnds1)
	# exit()
	# print("Time Diff: ",time.time()-t)
	# funcFit(linearFunc,xrand,yrand,example_classes,variance=None,method="BE",best_fit="average")
	# funcFit(linearFunc,xrand,yrand,example_classes,variance=None,method="BE",best_fit="peak1D")
	xrange=np.arange(0,12,0.1)
	plt.scatter(xrand,linearFunc(xrand,a,b,c),label="Data Points (Exact)")
	plt.scatter(xrand,yrand,label="Data Points (with Noise)")
	

	###GLS Fit
	plt.plot(xrange,linearFunc(xrange,a,**GLS_fit1),label="GLS Fit 1")
	
	###BE Fit
	plt.plot(xrange,linearFunc(xrange,a,**BE_fit1),label="BE Fit 1")
	plt.fill_between(xrange,*cont_bnds_funcFit(linearFunc,xrange,fit_param_combos=BE_bnds1["2.0 $\\sigma$"],linear_paramNames=[],a=a),label="BE Fit Bounds no-lin",alpha=0.5)
	plt.fill_between(xrange+1,*cont_bnds_funcFit(linearFunc,xrange,fit_param_combos=BE_bnds1["2.0 $\\sigma$"],linear_paramNames=["b"],a=a),label="BE Fit Bounds b-lin",alpha=0.3)
	plt.fill_between(xrange+2,*cont_bnds_funcFit(linearFunc,xrange,fit_param_combos=BE_bnds1["2.0 $\\sigma$"],linear_paramNames=["c"],a=a),label="BE Fit Bounds c-lin",alpha=0.3)
	plt.fill_between(xrange+3,*cont_bnds_funcFit(linearFunc,xrange,fit_param_combos=BE_bnds1["2.0 $\\sigma$"],linear_paramNames=["b","c"],a=a),label="BE Fit Bounds b,c-lin",alpha=0.3)
	# plt.fill_between(xrange,*cont_bnds_funcFit(linearFunc,xrange,fit_param_combos=BE_bnds1["2.0 $\\sigma$"],linear_paramNames=["b","c"],a=a),label="BE Fit Bounds b,c,d-lin",alpha=0.2)
	
	plt.legend()
	plt.show()
	plt.close()
	exit()

	theta_bnds=[[1,2],[2,3],[3,4],[4,5]]
	print(theta_bnds)
	r_0=3
	slopes=np.arange(0.4,5.01,0.1)
	slope=2
	Amps=np.arange(0.5,5.51,0.01)
	A_k=3
	# print(Amps)
	# print(r_0)
	# print(slopes)
	y_vals=projAngularAvg_KingOnly_gaussianConvolved(theta_bnds,1,r_0,slope,A_k,beam)
	# print(y_vals)
	print(y_vals.shape)
	classes=[func_param("r_0",r"r$_0$",r_0),func_param("slope",r"$\gamma$",slopes),func_param("z",r"z",1),func_param("A_k",r"A$_k$",Amps,is_linear=True),func_param("beam_fwhm",r"fwhm",2.10)]
	# BE_fit,BE_cov,BE_errs=funcFit(projAngularAvg_KingOnly_gaussianConvolved,theta_bnds,y_vals,classes,variance=None,method="BE",cornerplot=True,output_bnds=[1],plot_fit=False)[:3]
	# print(BE_fit)
	# print(BE_cov)
	# print(BE_errs)
