from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as spi
from scipy.optimize import curve_fit
import pandas as pd
import time



def angdiff(RA1, DEC1, RA2, DEC2, degrees=False):
	"""angle betwix locations.  Set degrees=True if input/output are in degrees"""
	if degrees:
		RA1 = np.deg2rad(RA1); DEC1=np.deg2rad(DEC1); RA2=np.deg2rad(RA2); DEC2=np.deg2rad(DEC2)
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

def df_cat_within(df, ra_mins, ra_maxs, dec_mins, dec_maxs, col_start=0):	###Grabs subset of catalog within a given RA and DEC range
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
		new_df = df[np.any(bnd_bools, axis=0)]
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

def change_stamp_gaussian_beam(stamp, pix_res, map_beam_fwhm, new_beam_fwhm, fft_cut_both=True, cut_val=0.25):
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

def remove_gradient(image, pix_res=0.05, center_cut_dist=0):
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

	image = np.array(image)	###To be safe
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
	avg_x = np.nanmean(mask_image, axis=0)
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
	avg_y = np.nanmean(mask_image, axis=1)
	x_matrix = np.column_stack([Y[:,0], np.ones_like(avg_y)])
	y_vector = np.array([avg_y,]).T
	N = len(y_vector)
	fit = np.linalg.inv(x_matrix.T @ x_matrix) @ x_matrix.T @ y_vector
	# scale = (y_vector - x_matrix@fit).T @ (y_vector - x_matrix@fit) / (N - len(fit))
	# fitcov=np.linalg.inv(x_matrix.T @ x_matrix) * scale
	# print("Y fit and cov: ",fit,fitcov)
	image = image - (Y*fit[0] + fit[1])
	return image

def radial_avg(stamp, rad_avg_radii, pix_res=0.05, pix_scale=False):
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

def tophat_aperture_sum(stamp, pix_res, th_rad, pix_scale=True):
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
def cap_circular_aperture_sum(stamp, pix_res, cap_rad, pix_scale=True):
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

def stamp_noise(stamp, pix_res, inner_drop_rad=0, outer_drop_rad=None):
	"""Calculates the noise of pixels for a given stamp, excluding pixels within an inner_drop_rad.  
	An outer_drop_rad can also be considered.  pix_res, inner_drop_rad, and outer_drop_rad should be in the same units"""

	DR = len(stamp)
	kern = np.full([DR, DR], np.nan)
	X, Y = np.meshgrid(np.arange(DR), np.arange(DR))
	dist = np.sqrt((X - int(np.rint((DR-1) / 2)))**2 + (Y - int(np.rint((DR-1) / 2)))**2) * pix_res
	
	if outer_drop_rad is None:    ###if no outer radius, compute the corners of the square too
		kern[dist > inner_drop_rad] = 1
	else:
		kern[(dist > inner_drop_rad) & (dist <= outer_drop_rad)] = 1
	
	return np.nanstd(stamp * kern)

### Here is code used in my old iterative high-pass filter/gaussian method, fitting and removing while trying to keep any central signal... not great, but in case anyone was curious
#   Used in my first Sunyaev-Zel'dovich paper: https://arxiv.org/abs/2103.01245
def g_fit(stamp, pix_res, fwhm, fwhm_max):
	"""Gaussian fitting of stamp's central source.  Used in old iterative gaussian filter procedure"""
	
	DR = len(stamp)
	c = int((DR-1) / 2)
	RAD = c * pix_res
	BASE = np.arange(-RAD, RAD + pix_res, pix_res) #Base array for stack mesh creation
	#	print(BASE)
	MESH1, MESH2 = np.meshgrid(BASE, BASE)
	xdata = np.vstack((MESH1.ravel(), MESH2.ravel()) )
	f2s = 1 / (2 * np.sqrt(np.log(4)) )
	
	def gaussian(x, y, peak,sig):
		# peak=args[0]
		# sig=args[1]
		return peak * np.exp(-((x)**2+(y)**2)/2/sig**2)

	def g(M,peak,sig):
		x, y = M
		sol = np.zeros(x.shape)
		sol += gaussian(x,y,peak,sig)
		return sol

	##Initial guess:
	p0 = (stamp[c,c], fwhm*f2s)
	bnds=([-np.inf,0.01],[np.inf,fwhm_max*f2s])
	popt, pcov = curve_fit(g, xdata, stamp.ravel(), p0=p0, bounds=bnds)[:2]
	
	return popt, pcov, gaussian(MESH1,MESH2,*popt)

def iGauss(stamp, pix_res, high_fwhm, beam_fwhm, verbose=True, return_inverse=False):
	"""Iterative High-Pass Gaussian fitting of a stamp.  High pass filters the stamp, fits a central gaussian to the resultant image, 
	then removes said gaussian fit from the original stamp and repeats again until no good central gaussian can be made."""

	i = 0
	t = 0
	f2s = 1 / (2 * np.sqrt(np.log(4)))	
	high_sig = f2s * high_fwhm / pix_res
	to_filter = stamp
	if verbose:
		print("	Result Std | Fit (peak, sig)		| Peak Fit S/N	| Sig Fit S/N	|")
	while i == 0:
		t += 1
		filter = spi.gaussian_filter(to_filter, sigma=high_sig)
		popt, pcov, fit = g_fit(to_filter - filter, pix_res, beam_fwhm, 1.5*beam_fwhm)
		std = stamp_noise(to_filter - filter, pix_res, 2*beam_fwhm, (len(stamp)-1)/2*pix_res - 2*beam_fwhm)[0]*10**-6		###The 1e-6 here from micro-kelvin conversion
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

def single_stack_plot(
		stamp, pix_res, save_name=None, sub_title=None, circle_arcs=[], 
		cbar_bounds=[0, 0], cbar_unit=r"$\mu $K$_{\rm CMB}$", cbar_dec_place=1, rasterized_plot=True):
	"""plot stamp cutout, either to show or to save if a given save_name is provided.  
	Make sure to include file extension, i.e. .png or .pdf in save_name
	
	Parameters
	----------
	stamp: 2d array
		Stamp cutout to plot.  Should be symmetric N x N array
	pix_res: float
		Pixel resolution of stamp.  Should be in units of arcmin
	save_name: str, optional
		If provided, save name to save the plot as.  Include full path and extension (.png, .pdf, etc)
	sub_title: str, optional
		Sub-title to write above plot.  Default == None
	circle_arc: list, optional
		1D list of radii in arcmin to sketch dashed circles at.  I.e. [2, 10] would create circles on the plot at radii of 2, and 10 arcmin
	cbar_bounds: list, optional
		Colorbar bounds [min, max].  If left at default [0, 0], it will infer from max and min of stamp
	cbar_unit: str, optional
		Units to write above colorbar. Default == r"$\mu $K$_{\rm CMB}$"

	Returns
	-------
	None, will either plot or save plot of stamp.
	"""
	stamp = np.array(stamp)	###To be safe
	DR = len(stamp)
	c = int((DR-1) / 2)
	RAD = c * pix_res
	BASE = np.arange(-RAD, RAD + pix_res, pix_res) #Base array for stack mesh creation
	BASE[int((DR)/2-1)] = 0
	# MESH1, MESH2 = np.meshgrid(BASE - pix_res/2, BASE - pix_res/2)
	fig, axs = plt.subplots(1, 1, sharex=False, sharey=True, figsize=(6,6))
	
	min=np.min(stamp)
	max=np.max(stamp)
	if cbar_bounds == [0, 0]:
		###pcolormesh has some annoying warnings, etc. so I'm just switching to imshow for now
		# mp = axs.pcolormesh(MESH1, MESH2, stamp, cmap="jet", vmin=min, vmax=max, rasterized=rasterized_plot)#, shading="nearest")
		mp = axs.imshow(stamp, cmap="jet", vmin=min, vmax=max, extent=[-RAD, RAD, -RAD, RAD], rasterized=rasterized_plot)
	else:
		# mp = axs.pcolormesh(MESH1, MESH2, stamp, cmap="jet", vmin=cbar_bounds[0], vmax=cbar_bounds[1], rasterized=rasterized_plot)#, shading="auto")
		mp = axs.imshow(stamp, cmap="jet", vmin=cbar_bounds[0], vmax=cbar_bounds[1], extent=[-RAD, RAD, -RAD, RAD], rasterized=rasterized_plot)	

	for c in range(len(circle_arcs)):
		circle1 = plt.Circle((0, 0), circle_arcs[c], fill=False, color="k", ls="--", lw=0.5)
		axs.add_patch(circle1)
	
	axs.set(aspect="equal")
	axs.title.set_text(sub_title)
	axs.set_xlabel(r"$\Delta\phi$ [arcmin]")
	axs.invert_xaxis()
	axs.invert_yaxis()
	axs.set_ylabel(r"$\Delta\theta$ [arcmin]")
	fig.colorbar(mp, ax=axs, label=cbar_unit, shrink=0.8, format="%."+"%if"%cbar_dec_place)
	if save_name:
		plt.savefig(save_name, dpi=400, bbox_inches=True)
		print("Plot saved as: " + save_name)
	else:
		plt.show()
	return None

def multi_stack_plot(
		stamps, pix_res, save_name=None, sub_titles=None, circle_arcs=[], 
		cbar_bounds=[0, 0], cbar_unit=r"$\mu $K$_{\rm CMB}$", cbar_dec_place=1, rasterized_plot=True):
	"""plot a row of stamp cutouts, either to show or to save if a given save_name is provided.  
	Make sure to include file extension, i.e. .png or .pdf in save_name
	
	Parameters
	----------
	stamp: list or np.ndarray
		Either a list or 3D array of stamp cutouts to plot.  First dimension should correspond to each stamp.  Each stamp should be symmetric N x N array (same size)
	pix_res: float
		Pixel resolution of stamps.  Should be in units of arcmin
	save_name: str, optional
		If provided, save name to save the plot as.  Include full path and extension (.png, .pdf, etc)
	sub_titles: str, optional
		Sub-titles to write above plot.  Default == None.  If provided, should have length == number of stamps given
	circle_arc: list, optional
		1D list of radii in arcmin to sketch dashed circles at.  I.e. [2, 10] would create circles on the plot at radii of 2, and 10 arcmin
	cbar_bounds: list, optional
		Colorbar bounds [min, max].  If left at default [0, 0], it will infer from max and min of stamp
	cbar_unit: str, optional
		Units to write above colorbar. Default == r"$\mu $K$_{\rm CMB}$"
	cbar_dec_place: int, optional
		Default == 1.  Number of decimal places for colorbar to display with
	rasterized_plot: bool, optional
		Default == True.  Whether to rasterize the plot (not text).  This helps reduce file size, even for pdfs

	Returns
	-------
	None, will either plot or save plot of row of stamps.
	"""
	vlen = len(stamps)
	DR = len(stamps[0])
	c = int((DR-1) / 2)
	RAD = c * pix_res
	BASE = np.arange(-RAD, RAD + pix_res, pix_res) #Base array for stack mesh creation
	BASE[int((DR)/2-1)] = 0
	# MESH1, MESH2 = np.meshgrid(BASE - pix_res/2, BASE - pix_res/2)

	fig, axs = plt.subplots(1, vlen, sharex=False, sharey=True, figsize=(2 + 4*vlen, 4))
	mins=[]
	maxs=[]
	for v in range(vlen):
		mins.append(np.min(stamps[v]))
		maxs.append(np.max(stamps[v]))
	for v in range(vlen):
		if cbar_bounds == [0,0]:
			# mp = axs[v].pcolormesh(MESH1, MESH2, stamps[v], cmap="jet", vmin=np.min(mins), vmax=np.max(maxs), rasterized=rasterized_plot, shading="auto")
			mp = axs[v].imshow(stamps[v], cmap="jet", vmin=np.min(mins), vmax=np.max(maxs), extent=[-RAD, RAD, -RAD, RAD], rasterized=rasterized_plot)
		else:
			# mp = axs[v].pcolormesh(MESH1, MESH2, stamps[v], cmap="jet", vmin=cbar_bounds[0], vmax=cbar_bounds[1], rasterized=rasterized_plot, shading="auto")
			mp = axs[v].imshow(stamps[v], cmap="jet", vmin=cbar_bounds[0], vmax=cbar_bounds[1], extent=[-RAD, RAD, -RAD, RAD], rasterized=rasterized_plot)	

		for c in range(len(circle_arcs)):
			circle1 = plt.Circle((0, 0), circle_arcs[c], fill=False, color="k", ls="--", lw=0.5)
			axs[v].add_patch(circle1)
		
		axs[v].set(aspect="equal")
		axs[v].title.set_text(sub_titles[v])
		axs[v].set_xlabel(r"$\Delta\phi$ [arcmin]")
	###Do this after everything?
	for v in range(vlen):
		axs[v].invert_xaxis()		
	axs[0].invert_yaxis()
	axs[0].set_ylabel(r"$\Delta\theta$ [arcmin]")

	fig.colorbar(mp, ax=axs[:], label=cbar_unit, shrink=0.8, format="%."+"%if"%cbar_dec_place)

	if save_name:
		plt.savefig(save_name, dpi=400, bbox_inches="tight")
		print("Plot saved as: " + save_name)
	else:
		plt.show()
	return None


def multi_stack_grid_plot(
		stamps, pix_res, col_titles, row_titles, save_name=None, circle_arcs=[],
		cbar_bounds=[0, 0], cbar_unit=r"$\mu $K$_{\rm CMB}$", cbar_dec_place=1, rasterized_plot=True):
	"""Where stamps must be a 1-2d list or array of equal-sized 2d stamp cutout arrays. Each row must have the same length.
	
	Parameters
	----------
	stamp: list or np.ndarray
		Either a list or 3D array of stamp cutouts to plot.  First dimension should correspond to each stamp.  Each stamp should be symmetric N x N array (same size)
	pix_res: float
		Pixel resolution of stamps.  Should be in units of arcmin
	col_titles: list
		Titles for each stamp column.  Should be equal length to number of stamp columns
	row_titles: list
		Titles for each stamp row.  Should be equal length to number of stamp rows
	save_name: str, optional
		If provided, save name to save the plot as.  Include full path and extension (.png, .pdf, etc)
	circle_arc: list, optional
		1D list of radii in arcmin to sketch dashed circles at.  I.e. [2, 10] would create circles on the plot at radii of 2, and 10 arcmin
	cbar_bounds: list, optional
		Colorbar bounds [min, max].  If left at default [0, 0], it will infer from max and min of stamp
	cbar_unit: str, optional
		Units to write above colorbar. Default == r"$\mu $K$_{\rm CMB}$"
	cbar_dec_place: int, optional
		Default == 1.  Number of decimal places for colorbar to display with
	rasterized_plot: bool, optional
		Default == True.  Whether to rasterize the plot (not text).  This helps reduce file size, even for pdfs

	Returns
	-------
	None, will either plot or save plot of row of stamps.
	"""
	
	stamps = np.asarray(stamps)
	if len(vshape) == 3:	###i.e. 1d list/array of images (2d)
		stamps = stamps[None, ...]
	vshape = stamps.shape
	DR = len(stamps[0,0])
	c = int((DR-1) / 2)
	RAD = c * pix_res
	BASE = np.arange(-RAD, RAD + pix_res, pix_res) #Base array for stack mesh creation
	BASE[int((DR)/2-1)] = 0
	MESH1, MESH2 = np.meshgrid(BASE - pix_res/2, BASE - pix_res/2)
	
	###Trying to scale stuff... decently?  (For published plots I often copy this function and customize it a bit further)
	factor = 4
	gridxscale = factor * vshape[1] + factor/2
	gridyscale = factor * vshape[0] + 1
	
	fig, axs = plt.subplots(vshape[0], vshape[1], figsize=(gridxscale, gridyscale), sharex="col", sharey="row")
	
	mins=[]
	maxs=[]
	###Setting colorbar range
	for i in range(vshape[0]):
		for j in range(vshape[1]):
			mins.append(np.min(stamps[i, j]))
			maxs.append(np.max(stamps[i, j]))

	if cbar_bounds == [0, 0]:
		c_min = np.min(mins)
		c_max = np.max(maxs)
	else:
		c_min = cbar_bounds[0]
		c_max = cbar_bounds[1]

	for i in range(vshape[0]):
		for j in range(vshape[1]):
			if i == vshape[0]-1:
				axs[i,j].invert_xaxis()
				axs[i,j].set_xlabel(r"$\Delta\phi$ [arcmin]")
				axs[i,j].set_xlim(-RAD, RAD)
			if j == 0:
				axs[i,j].invert_yaxis()
				axs[i,j].set_ylabel(r"$\Delta\theta$ [arcmin]")

			# mp = axs[i,j].pcolormesh(MESH1, MESH2, stamps[i,j], cmap="jet", vmin=c_min, vmax=c_max, rasterized=rasterized_plot, shading="auto")
			mp = axs[i,j].imshow(stamps[i,j], cmap="jet", vmin=c_min, vmax=c_max, extent=[-RAD, RAD, -RAD, RAD], rasterized=rasterized_plot)	
			for c in range(len(circle_arcs)):
				circle1 = plt.Circle((0, 0), circle_arcs[c], fill=False, color="k", ls="--", lw=0.5)
				axs[i,j].add_patch(circle1)
			axs[i,j].set(aspect="equal")
			axs[0,j].set_title(col_titles[j])
		axs[i,0].text(-0.65 * vshape[0] / (factor+1), 0.5, row_titles[i], verticalalignment="center", size=12, rotation=90, transform=axs[i,0].transAxes)
	
	plt.subplots_adjust(left=(factor+1) * 0.24 / gridxscale, bottom=(factor+1) * 0.3 / gridxscale, hspace=(factor+1) * 0.22 / gridyscale, wspace=(factor+1) * 0.1 / gridxscale)
	cbar_ax = fig.add_axes([0.885 + (factor+1)*0.06/gridxscale, 0.18, (factor+1) * 0.04 / gridyscale, 0.6])
	fig.colorbar(mp, ax=axs[:,:], cax=cbar_ax, format="%."+"%if"%cbar_dec_place)
	cbar_ax.set_title(cbar_unit)

	if save_name:
		plt.savefig(save_name, dpi=400, bbox_inches="tight")
		print('Plot saved as: ' + save_name)
	else:
		plt.show()
	return None

def corr_plot(
		corr, x_set, x_skip=4, x_dec=2, axis_label=None, title=None, shift=0., 
		custom_cmap=None, cbar_lims=[-1, 1], cbar_ticks=[-1, -.5, 0, .5, 1], save_name=None, fsize=14, smaller_fsize=11, rasterized_plot=True):
	"""My quick correlation matrix plot maker.  Takes 2d correlation array and plots it.
	
	Parameters
	----------
	corr: 2d numpy.ndarray
		correlation matrix
	x_set: 1d list or array
		axis values along both x and y
	x_skip: int, optional
		how many x_set values/ticks should be labeled. Default == 4 implies every 4 ticks are labeled on axis
	x_dec: int, optional	
		decimal places for x_set tick labels on axis.  Default == 2
	axis_label: str, optional
		string to label axes
	title: str, optional
		string to label top of plot
	shift: float, optional
		shifting of ticks, useful if you wish center of squares to match with ticks instead of edges (say, shift = 0.5 for ticks spaced by 1)
	custom_cmap: matplotlib.colormap
		custom colormap able to be passed through.  Else default is reversed "RdYlBu"
	cbar_lims: list or numpy.ndarray, optional
		Two element 1d list or array of colorbar limits to use.  Default == [-1, 1] is standard range
	cbar_ticks: list or numpy.ndarray, optional
		Used to specify colorbar ticks to avoid crowded visuals.  Default == [-1, -.5, 0, .5, 1]
	save_name: str, optional
		If provided, save name to save the plot as.  Include full path and extension (.png, .pdf, etc)
	fsize: float, optional
		Main font size used in axis labels.  Default == 14
	smaller_fsize: float, optional
		Font size used in the smaller text, i.e. tick labels, etc. Default == 11
	rasterized_plot: bool, optional
		Default == True.  Whether to rasterize the plot (not text).  This helps reduce file size, even for pdfs

	Returns
	-------
	None, will either plot or save plot of correlation matrix.

	"""


	plt.rc("xtick", labelsize=smaller_fsize)
	plt.rc("ytick", labelsize=smaller_fsize)
	plt.minorticks_on()
	if custom_cmap:
		plt.imshow(corr, origin="lower", cmap=custom_cmap, vmin=cbar_lims[0], vmax=cbar_lims[1], rasterized=rasterized_plot)
	else:
		plt.imshow(corr, origin="lower", cmap=plt.cm.get_cmap("RdYlBu").reversed(), vmin=cbar_lims[0], vmax=cbar_lims[1], rasterized=rasterized_plot)
	if x_dec == 0:
		plt.xticks(np.arange(len(x_set))[:: x_skip] - shift, labels=np.array(np.round(x_set[:: x_skip], x_dec),dtype=int))
		plt.yticks(np.arange(len(x_set))[:: x_skip] - shift, labels=np.array(np.round(x_set[:: x_skip], x_dec),dtype=int))
	else:
		plt.xticks(np.arange(len(x_set))[:: x_skip] - shift, labels=np.round(x_set[:: x_skip], x_dec))
		plt.yticks(np.arange(len(x_set))[:: x_skip] - shift, labels=np.round(x_set[:: x_skip], x_dec))
	if axis_label:
		plt.xlabel(axis_label, fontsize=fsize)
		plt.ylabel(axis_label, fontsize=fsize)
	
	plt.colorbar(ticks=cbar_ticks)
	if title:
		plt.title(title, fontsize=fsize)
	if save_name:
		plt.savefig(save_name, dpi=400, bbox_inches="tight")
		print("Plot saved as: " + save_name)
	else:
		plt.tight_layout()
		plt.show()
	plt.close()
	return None

if __name__=="__main__":    ###For any other quick testing purposes
	print("Start Time: ", time.asctime())


	
	print("End Time: ", time.asctime())