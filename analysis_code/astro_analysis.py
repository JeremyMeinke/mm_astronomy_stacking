from __future__ import division
import numpy as np
import healpy as hp
import scipy.integrate as spint
import scipy.ndimage as spi
import scipy.special as spsp
from scipy.optimize import curve_fit
from astropy.cosmology import LambdaCDM
import scipy.special as spsp
import time
from typing import Any, Callable

###tSZ and dust stuff
def cmb_temp_to_intensity(freq, T_CMB=2.725):		###Intensity in [Jy or Jy*sr if solid angle was left in]
	"""Converts 1 [Kelvin] in T_CMB units into [Jy], or [Jy*sr] if solid angle is still left integrated
	
	Parameters
	----------
	freq: float
		Frequency in [GHz]
	T_CMB: float, optional
		Default = 2.725 [Kelvin]
	
	Returns
	-------
	float (Intensity value at given frequency)
	"""
	k = 1.38*10**(-23) #[J/K]
	h = 6.626*10**(-34) #[J*s]
	c = 2.998*10**8		#[m/s]
	x = freq * 10**9 * h / k / T_CMB
	return 10**(26) * 2 * (k*T_CMB)**3 / (h*c)**2 / T_CMB * x**4 * np.exp(x) / (np.exp(x)-1)**2

def bb_spectrum(freq, T, a=1., z=0., modified=False, beta=0., v_0=353.):
	"""Blackbody and modified blackbody spectrum
	
	Parameters
	----------
	freq: float
		Frequency in [GHz]
	T: float
		Temperature of blackbody
	a: float, optional
		Amplitude to scale by, Default == 1
	z: float, optional
		Redshift, Default == 0
	modified: bool, optional
		If modified Blackbody, Default == False
	beta: float, optional
		Emissivity index, used if modified == True
	v_0: float, optional
		Reference frequency [GHz] of mbb, used if modified == True
	
	Returns
	-------
	float (Blackbody value at given frequency)
	"""
	k = 1.38*10**(-23) #[J/K]
	h = 6.626*10**(-34) #[J*s]
	c = 2.998*10**8		#[m/s]
	x = (1+z) * freq * 10**9 * h / k / T
	if modified: ###Modified Blackbody
		x_0 = v_0 * 10**9 * h / k / T
		return a * 2 * (k*T)**3 / (h*c)**2 * x**3 / (np.exp(x)-1) * ((x/x_0)**beta)
	else:
		return a * 2 * (k*T)**3 / (h*c)**2 * x**3 / (np.exp(x)-1)

def y_to_mJy(freq, y, arcmin=True):	###Converts y back into T_CMB, then to intensity
	"""Compton y through a solid angle, converted into milli-Janskies [mJy]
	arcmin=True for when measurement is over an arcmin^2 solid angle instead of steradian
	
	Parameters
	----------
	freq: float
		Frequency in [GHz]
	y: float
		Integrated Compton-y amplitude [sr or arcmin^2] to scale by
	arcmin: bool, optional
		If integrated solid angle is arcmin^2 instead of steradian (rad^2)
	
	Returns
	-------
	float (Integrated Compton-y in mJy)
	"""
	T_CMB = 2.725 #K
	k = 1.38*10**(-23) #[J/K]
	h = 6.626*10**(-34) #[J*s]
	x_freq = freq * 10**9 * h / k / T_CMB
	if arcmin:
		arcmin_to_radian = np.pi / 180 / 60		#[rad/arcmin]
		return y * T_CMB * (x_freq*(np.exp(x_freq)+1)/(np.exp(x_freq)-1)-4) * cmb_temp_to_intensity(freq) * 1e3 * arcmin_to_radian**2
	else:
		return y * T_CMB * (x_freq*(np.exp(x_freq)+1)/(np.exp(x_freq)-1)-4) * cmb_temp_to_intensity(freq) * 1e3

def dust_intensity_to_mJy(freq, dust_intensity, T_dust, beta_dust, z, v_0=353, arcmin=True):
	"""Dust intensity integrated over solid angle [W / Hz / m^2] in REFERENCE FRAME to milli-Janskies [mJy] in OBSERVED FRAME
	Note: reference to observed includes an additional (1+z) / (1+z)**2 term
	arcmin=True for when measurement is over an arcmin^2 solid angle
	
	Parameters
	----------
	freq: float
		Frequency in [GHz]
	dust_intensity: float
		Integrated dust intensity amplitude [W / Hz / m^2] to scale by
	T_dust: float
		Dust temperature
	beta_dust: float
		Dust emissivity index
	z: float
		Redshift
	v_0: float, optional
		Reference frequency [GHz]
	arcmin: bool, optional
		If integrated solid angle is arcmin^2 instead of steradian (rad^2)
	
	Returns
	-------
	float
	"""
	if arcmin:
		arcmin_to_radian = np.pi / 180 / 60		#[rad/arcmin]
		return dust_intensity / (1+z) * bb_spectrum(freq, T_dust, z=z, modified=True, beta=beta_dust, v_0=v_0) / bb_spectrum(v_0, T_dust, z=0, modified=True, beta=beta_dust, v_0=v_0) * 1e29 * arcmin_to_radian**2
	else:
		return dust_intensity / (1+z) * bb_spectrum(freq, T_dust, z=z, modified=True, beta=beta_dust, v_0=v_0) / bb_spectrum(v_0, T_dust, z=0, modified=True, beta=beta_dust, v_0=v_0) * 1e29

def uK_to_mJy(freq, uK, arcmin=True):
	"""convert micro Kelvin measurement (uK) to milli-Janskies (mJy) at the given frequency [GHz]
	arcmin=True for when measurement is over an arcmin^2 solid angle
	"""
	###1e-3 term from uK and mJy (1e3/1e-6)
	if arcmin:
		arcmin_to_radian = np.pi / 180 / 60		#[rad/arcmin]
		return uK * cmb_temp_to_intensity(freq) * 1e-3 * arcmin_to_radian**2
	else:
		return uK * cmb_temp_to_intensity(freq) * 1e-3

def tsz_dust_single_freq(freq, z, beta_dust, t_dust, a_y, a_dust, dust_v_0=353, T_CMB=2.725):
	"""Where freq == 2 rows: frequency band [row 0] and weights [row 1].  dust_v_0 is dust reference/rest frequency in GHz to scale to.
	
	Parameters
	----------
	freq: float or np.ndarray
		frequency to measure/sample/calculate at
	z: float
		Redshift of sample
	beta_dust: float
		Dust emissivity index (Beta)
	t_dust: float
		Dust temperature
	a_y: float
		Compton-y amplitude (thermal Sunyaev Zel-dovich effect)
	a_dust: float
		Dust amplitude (in rest frequency)
	dust_v_0: float
		Dust rest frequency to scale to, in [GHz] (Default == 353)
	T_CMB: float, optional
		Default = 2.725 [Kelvin]
	
	Returns
	-------
	float, in units of K_CMB integrated over the observed frequency band
	"""
	
	
	###Other Constants
	k = 1.38*10**(-23) 	#[J/K]
	h = 6.626*10**(-34) #[J*s]
	c = 2.99792458e8	#[m/s]

	x_freq = freq * 10**9 * h / k / T_CMB

	def dust_func(v_f):	### v_f in [GHz]
		x_f=v_f*10**9*h/k/t_dust	###Unitless frequency
		###Now scaling by dust_v_0 GHz as reference frequency of gray-body.  Making sure to including constant term in front for safety...
		return 2 * (k*t_dust)**3 / (h*c)**2 * (v_f/dust_v_0)**beta_dust * (x_f)**3 / (np.exp(x_f)-1)
	
	def dBdT_func(x_f):		###In T_CMB units, REMEMBER THE CONSTANT TERM OUT IN FRONT....
		return 2 * k * (k*T_CMB)**2 / (h*c)**2 * (x_f)**4 * np.exp(x_f) / (np.exp(x_f)-1)**2
	
	###Compton-y weighted averages
	y_int = T_CMB * (x_freq*(np.exp(x_freq)+1) / (np.exp(x_freq)-1) - 4)
	# y_int=1	###When testing
	d_int = dust_func(freq * (1+z)) / dBdT_func(x_freq) / (1+z) / dust_func(dust_v_0)			####Now factor of (1+z)/(1+z)^2
	# Dx_int=1	###When testing
	return a_y*y_int + a_dust*d_int

def tsz_dust_over_band(freq, z, beta_dust, t_dust, a_y, a_dust, dust_v_0=353, T_CMB=2.725):
	"""Where freq == 2 rows: frequency band [row 0] and weights [row 1].  dust_v_0 is dust reference/rest frequency in GHz to scale to.
	
	Parameters
	----------
	freq: 2d array
		2d numpy array of observed frequency response (size 2 x N), where first index is frequency and second index is response
	z: float
		Redshift of sample
	beta_dust: float
		Dust emissivity index (Beta)
	t_dust: float
		Dust temperature
	a_y: float
		Compton-y amplitude (thermal Sunyaev Zel-dovich effect)
	a_dust: float
		Dust amplitude (in rest frequency)
	dust_v_0: float
		Dust rest frequency to scale to, in [GHz] (Default == 353)
	T_CMB: float, optional
		Default = 2.725 [Kelvin]

	Returns
	-------
	float, in units of K_CMB integrated over the observed frequency band
	"""
	
	
	###Other Constants
	k = 1.38*10**(-23) 	#[J/K]
	h = 6.626*10**(-34) #[J*s]
	c = 2.99792458e8	#[m/s]

	x_freq = freq[0] * 10**9 * h / k / T_CMB
	band=freq[1]

	def dust_func(v_f):	### v_f in [GHz]
		x_f=v_f*10**9*h/k/t_dust	###Unitless frequency
		###Now scaling by dust_v_0 GHz as reference frequency of gray-body.  Making sure to including constant term in front for safety...
		return 2 * (k*t_dust)**3 / (h*c)**2 * (v_f/dust_v_0)**beta_dust * (x_f)**3 / (np.exp(x_f)-1)
	
	def dBdT_func(x_f):		###In T_CMB units, REMEMBER THE CONSTANT TERM OUT IN FRONT....
		return 2 * k * (k*T_CMB)**2 / (h*c)**2 * (x_f)**4 * np.exp(x_f) / (np.exp(x_f)-1)**2
	
	# print(band.shape)
	band_sum = spint.trapz(band, x=freq[0], axis=0)	###Sum/Integral of frequency band weights to divide frequency spectrum by (when doing weighted avgs)

	###Compton-y weighted averages
	y_int = spint.simps(band * T_CMB * (x_freq*(np.exp(x_freq)+1) / (np.exp(x_freq)-1) - 4), x=freq[0], axis=0) / band_sum
	# y_int=1	###When testing
	d_int = spint.simps(band * dust_func(freq[0] * (1+z)) / dBdT_func(x_freq), x=freq[0], axis=0) / (1+z) / band_sum / dust_func(dust_v_0)			####Now factor of (1+z)/(1+z)^2
	# Dx_int=1	###When testing
	return a_y*y_int + a_dust*d_int


###Radial profile stuff
def proj_angular_avg_king_convolved(
		theta_bnds, z, r_0, slope, a_k, beam_fwhm, metric=LambdaCDM(H0=68, Om0=0.31, Ode0=0.69, Ob0=0.049, Tcmb0=2.725),
		dist="comoving", slope_like_powerlaw=True):
	"""Projected angular [arcmin] profile average for a unprojected king defined as k(r)=A_k*(1+(r/r_0)^2)^(-3*slope/2) between bounds of theta_bnds.
	slope_like_powerLaw=True makes the unprojected power of -gamma/2 (matching power law of -gamma), 
	instead of -3*gamma/2 (other standard convention resulting in gamma=1 for projected power equal to 1)
	
	Parameters
	---------
	theta_bnds: list or array
		Either list of lists of angular bounds to integrate within, or a 2d array where index 0 is lower bounds and 1 is upper bounds. In units of arcminutes
	z: float
		redshift of profile/sample
	r_0: float
		"Core" or reference radius to scale model by, in Mpc
	slope: float
		slope exponent in profile function
	a_k: float
		King amplitude
	beam_fwhm: float
		fwhm of beam to convolve by, in units of arcminutes
	metric: astropy.cosmology.LambdaCDM, optional
		LambdaCDM model to pass for calculating distances.  Default == LambdaCDM(H0=68, Om0=0.31, Ode0=0.69, Ob0=0.049, Tcmb0=2.725)
	dist: str, optional
		Distance calculation method, "comoving" or "proper" distance.  Default == "comoving"
	slope_like_powerlaw: bool, optional
		If slope parameter should be defined in similar manner to powerlaw in astronomy.  Default = True (see opening note for other way)

	Returns
	-------
	numpy.ndarray, of integrated values over profile
	
	"""
	###Setup
	sig = beam_fwhm / np.sqrt(8.*np.log(2))
	if dist.lower() == "comoving":
		D_arcmin = metric.comoving_transverse_distance(z).value * np.pi / 180 / 60
	elif dist.lower() == "proper":
		D_arcmin = metric.angular_diameter_distance(z).value * np.pi / 180 / 60
	else:
		print("dist not recognized... assuming you meant comoving...")
		D_arcmin = metric.transverse_comoving_distance(z).value * np.pi / 180 / 60
	
	if type(theta_bnds) == list:		###i.e. if list of bnds, each of len==2 (or just one list of 2 floats...)
		theta_bnds = np.stack(theta_bnds, axis=-1)
	###Defining inner integral function, amplitude now removing that r_0 term for simplicity. NOTE: i0e has a constant exp(-np.abs(x)) in front, hence why we have np.exp((-(theta-t)**2)/2/sig**2)
	if slope_like_powerlaw:
		king_gauss = lambda  t, theta: theta * a_k * spsp.gamma(1/2) * spsp.gamma((slope-1)/2) / spsp.gamma(slope/2) * (1+(D_arcmin*t/r_0)**2)**((1-slope)/2) / sig**(2) * t * np.exp((-(theta-t)**2)/2/sig**2) * spsp.i0e(theta*t/sig**2)
	else:	
		king_gauss = lambda  t, theta: theta * a_k * spsp.gamma(1/2) * spsp.gamma((3*slope-1)/2) / spsp.gamma(3*slope/2) * (1+(D_arcmin*t/r_0)**2)**((1-3*slope)/2) / sig**(2) * t * np.exp((-(theta-t)**2)/2/sig**2) * spsp.i0e(theta*t/sig**2)
	###Defining full integral function (double integral using quad_vec to allow for vectorizing of z, r_0, slope, A_k, and beam_fwhm)
	int_func = lambda b_lower, b_upper: 2 * np.pi * spint.quad_vec(lambda theta: spint.quad_vec(lambda x: king_gauss(x, theta), 0, np.inf,workers=1)[0], b_lower, b_upper, workers=1)[0]		###only care about integral return, not error (always is good from my separate testing)
	if type(theta_bnds[0]) != np.ndarray:		###Assume theta_bnds was given as 1D array or list
		return int_func(theta_bnds[0], theta_bnds[1]) / (rad_avg_area(theta_bnds[0], theta_bnds[1]))
	else:	###Should be 2D array with 2 rows (0==lower bnd, 1==upper)  Flattening for any larger arrays needed (*cough* curve fitting *cough*)
		t_lower = theta_bnds[0].flatten()
		t_upper = theta_bnds[1].flatten()
		return np.asarray([int_func(t_lower[i], t_upper[i]) / (rad_avg_area(t_lower[i], t_upper[i])) for i in range(len(t_lower))])		###2 pi factors cancel each other out

def proj_angular_avg_ptsrc_king_convolved(
		theta_bnds, z, r_0, slope, a_k, a_ps, beam_fwhm, c=0, metric=LambdaCDM(H0=68, Om0=0.31, Ode0=0.69, Ob0=0.049, Tcmb0=2.725),
		dist="comoving", beam_lmax=None, slope_like_powerlaw=True):
	"""Projected angular [arcmin] profile average for a projected king k(r)=A_k/r_0*(1+(r/r_0)^2)^(-3*slope/2) between bounds of theta_bnds.
	slope_like_powerLaw=True makes the unprojected power of -gamma/2 (matching power law of -gamma), 
	instead of -3*gamma/2 (other standard convention resulting in gamma=1 for projected power equal to 1)
	
	Parameters
	---------
	theta_bnds: list or array
		Either list of lists of angular bounds to integrate within, or a 2d array where index 0 is lower bounds and 1 is upper bounds. In units of arcminutes
	z: float
		redshift of profile/sample
	r_0: float
		"Core" or reference radius to scale model by, in Mpc
	slope: float
		slope exponent in profile function
	a_k: float
		King amplitude
	a_ps: float
		Point source amplitude
	beam_fwhm: float
		fwhm of beam to convolve by (and of point source), in units of arcminutes
	c: float, optional
		Constant term/offset (mainly for testing purposes) Default == 0
	metric: astropy.cosmology.LambdaCDM, optional
		LambdaCDM model to pass for calculating distances.  Default == LambdaCDM(H0=68, Om0=0.31, Ode0=0.69, Ob0=0.049, Tcmb0=2.725)
	dist: str, optional
		Distance calculation method, "comoving" or "proper" distance.  Default == "comoving"
	beam_lmax: int, optional
		If a lmax (max ell, spherical harmonic) cutoff for beam.  Used in point source creation (gaussian if None, otherwise uses healpy to make) Default == None
	slope_like_powerlaw: bool, optional
		If slope parameter should be defined in similar manner to powerlaw in astronomy.  Default = True (see opening note for other way)

	Returns
	-------
	numpy.ndarray, of integrated values over profile
	"""
	###Setup
	sig = beam_fwhm / np.sqrt(8.*np.log(2))
	arcmin_to_radian = np.pi / 180 / 60
	if dist.lower() == "comoving":
		D_arcmin = metric.comoving_transverse_distance(z).value * np.pi / 180 / 60
	elif dist.lower() == "proper":
		D_arcmin = metric.angular_diameter_distance(z).value * np.pi / 180 / 60
	else:
		print("dist not recognized... assuming you meant comoving...")
		D_arcmin = metric.comoving_transverse_distance(z).value * np.pi / 180 / 60
	
	if type(theta_bnds) == list:		###i.e. if list of bnds, each of len==2 (or just one list of 2 floats...)
		theta_bnds = np.stack(theta_bnds, axis=-1)
	# print(theta_bnds)
	###Defining inner integral function (extra theta term to be integrated in intfunc) NOTE: i0e has a constant exp(-np.abs(x)) in front, hence why we have np.exp((-(theta-t)**2)/2/sig**2)
	if slope_like_powerlaw:
		king_gauss = lambda  t, theta: theta * a_k * spsp.gamma(1/2) * spsp.gamma((slope-1)/2) / spsp.gamma(slope/2) * (1+(D_arcmin*t/r_0)**2)**((1-slope)/2) / sig**(2) * t * np.exp((-(theta-t)**2)/2/sig**2) * spsp.i0e(theta*t/sig**2)
	else:	
		king_gauss = lambda  t, theta: theta * a_k * spsp.gamma(1/2) * spsp.gamma((3*slope-1)/2) / spsp.gamma(3*slope/2) * (1+(D_arcmin*t/r_0)**2)**((1-3*slope)/2) / sig**(2) * t * np.exp((-(theta-t)**2)/2/sig**2) * spsp.i0e(theta*t/sig**2)
	###Defining full integral function (double integral using quad_vec to allow for vectorizing of z, r_0, slope, A_k, and beam_fwhm)
	if beam_lmax:
		beam_bl = hp.gauss_beam(beam_fwhm*arcmin_to_radian, beam_lmax)
		###Okay, new way, as quad doesn't like healpy or visa-versa...
		arc_steps = np.linspace(0, 50*sig, num=2000)	###For easier adjusting
		s_norm = 2 * np.pi * spint.simps(arc_steps*hp.bl2beam(beam_bl, arcmin_to_radian*arc_steps)) * (np.diff(arc_steps[:2]))	###Radial integral to 2d, accounting for discrete int. stepsize
		def int_func(b_lower, b_upper):
			rad_bin_steps=np.linspace(b_lower, b_upper, num=int(np.round((b_upper-b_lower)*50)))	###For easier adjusting
			return 2 * np.pi * (spint.quad_vec(lambda theta: spint.quad_vec(lambda x: king_gauss(x, theta), 0, np.inf)[0], b_lower, b_upper)[0] + a_ps*spint.simps(rad_bin_steps*hp.bl2beam(beam_bl,arcmin_to_radian*rad_bin_steps))*(np.diff(rad_bin_steps[:2]))/s_norm)
		###only care about integral return, not error (always is good from my separate testing).  
	else:
		int_func = lambda b_lower, b_upper: 2*np.pi*spint.quad_vec(lambda theta: spint.quad_vec(lambda x: king_gauss(x, theta), 0, np.inf)[0], b_lower, b_upper)[0] + a_ps*(np.exp(-(b_lower/sig)**2/2)-np.exp(-(b_upper/sig)**2/2))
	if type(theta_bnds[0]) != np.ndarray:		###Assume theta_bnds was given as 1D array or list
		return int_func(theta_bnds[0], theta_bnds[1]) / (rad_avg_area(theta_bnds[0], theta_bnds[1])) + c
	else:	###Should be 2D array with 2 rows (0==lower bnd, 1==upper)  Flattening for any larger arrays needed (*cough* curve fitting *cough*)
		t_lower = theta_bnds[0].flatten()
		t_upper = theta_bnds[1].flatten()
		return np.asarray([int_func(t_lower[i], t_upper[i]) / (rad_avg_area(t_lower[i], t_upper[i])) for i in range(len(t_lower))]) + c

def proj_angular_avg_ptsrc_convolved(theta_bnds, a_ps, beam_fwhm, c=0, beam_lmax=None):
	"""Projected angular [arcmin] profile average for a central point source. 
	*NOW* With optional additional uniform offset constant C added
	
	Parameters
	---------
	theta_bnds: list or array
		Either list of lists of angular bounds to integrate within, or a 2d array where index 0 is lower bounds and 1 is upper bounds. In units of arcminutes
	z: float
		redshift of profile/sample
	a_ps: float
		Point source amplitude
	beam_fwhm: float
		fwhm of beam to convolve by (and of point source), in units of arcminutes
	c: float, optional
		Constant term/offset (mainly for testing purposes) Default == 0
	beam_lmax: int, optional
		If a lmax (max ell, spherical harmonic) cutoff for beam.  Used in point source creation (gaussian if None, otherwise uses healpy to make) Default == None

	Returns
	-------
	numpy.ndarray, of integrated values over profile
	"""
	###Setup
	sig = beam_fwhm / np.sqrt(8.*np.log(2))
	arcmin_to_radian = np.pi / 180 / 60
	if type(theta_bnds) == list:		###i.e. if list of bnds, each of len==2 (or just one list of 2 floats...)
		theta_bnds = np.stack(theta_bnds, axis=-1)
	
	if beam_lmax:
		beam_bl = hp.gauss_beam(beam_fwhm*arcmin_to_radian, beam_lmax)
		###Okay, new way, as quad doesn't like healpy or visa-versa...
		arc_steps = np.linspace(0, 50*sig, num=2000)	###For easier adjusting
		s_norm = 2 * np.pi * spint.simps(arc_steps*hp.bl2beam(beam_bl, arcmin_to_radian*arc_steps))*(np.diff(arc_steps[:2]))	###Radial integral to 2d, accounting for discrete int. stepsize
		def func(b_lower, b_upper):
			rad_bin_steps = np.linspace(b_lower, b_upper, num=int(np.round((b_upper-b_lower)*50)))	###For easier adjusting
			return 2 * np.pi * (a_ps*spint.simps(rad_bin_steps*hp.bl2beam(beam_bl, arcmin_to_radian*rad_bin_steps))*(np.diff(rad_bin_steps[:2]))/s_norm)
		###only care about integral return, not error (always is good from my separate testing).  
	else:
		func = lambda b_lower, b_upper: a_ps * (np.exp(-(b_lower/sig)**2/2) - np.exp(-(b_upper/sig)**2/2))
	if type(theta_bnds[0]) != np.ndarray:		###Assume theta_bnds was given as 1D array or list
		return func(theta_bnds[0], theta_bnds[1]) / (rad_avg_area(theta_bnds[0], theta_bnds[1])) + c
	else:	###Should be 2D array with 2 rows (0==lower bnd, 1==upper)  Flattening for any larger arrays needed (*cough* curve fitting *cough*)
		t_lower = theta_bnds[0].flatten()
		t_upper = theta_bnds[1].flatten()
		return np.asarray([func(t_lower[i], t_upper[i]) / (rad_avg_area(t_lower[i], t_upper[i])) for i in range(len(t_lower))])+c

def proj_ptsrc_convolved(thetas, a_ps, beam_fwhm, beam_lmax=None):
	""" Point source profile when convolved with beam.  Allows for lmax cut that introduces non-gaussian side lobes/effects
	
	Parameters
	---------
	thetas: list or array
		Either list or array of thetas to sample profile at.  In units of arcminutes
	a_ps: float
		Point source amplitude
	beam_fwhm: float
		fwhm of beam to convolve by (and of point source), in units of arcminutes
	beam_lmax: int, optional
		If a lmax (max ell, spherical harmonic) cutoff for beam.  Used in point source creation (gaussian if None, otherwise uses healpy to make) Default == None

	Returns
	-------
	numpy.ndarray, of integrated values over profile, same shape as thetas provided
	"""

	sig = beam_fwhm / np.sqrt(8.*np.log(2))
	thetas = np.array(thetas)
	thetas_shape = thetas.shape
	thetas = np.ravel(thetas)
	if beam_lmax:
		arcmin_to_radian = np.pi / 180 / 60
		beam_bl = hp.gauss_beam(beam_fwhm*arcmin_to_radian, beam_lmax)
		theta_steps = np.linspace(0, 25*sig, num=2000)
		s_norm = 2 * np.pi * np.diff(theta_steps[:2]) * spint.simps(theta_steps*hp.bl2beam(beam_bl, arcmin_to_radian*theta_steps))
		return a_ps * (hp.bl2beam(beam_bl, thetas*arcmin_to_radian)/s_norm).reshape(thetas_shape)
	else:
		return a_ps / (2*np.pi*sig**2) * np.exp(-(thetas.reshape(thetas_shape)/sig)**2/2)

def proj_king_convolved(thetas, z, r_0, slope, a_k, beam_fwhm, metric=LambdaCDM(H0=68,Om0=0.31,Ode0=0.69,Ob0=0.049,Tcmb0=2.725), dist="comoving", slope_like_powerlaw=True):
	"""slope_like_powerLaw=True makes the unprojected power of -gamma/2 (matching power law of -gamma), 
	instead of -3*gamma/2 (other standard convention resulting in gamma=1 for projected power equal to 1)
	
	Parameters
	---------
	thetas: list or array
		Either list or array of thetas to sample profile at.  In units of arcminutes
	z: float
		redshift of profile/sample
	r_0: float
		"Core" or reference radius to scale model by, in Mpc
	slope: float
		slope exponent in profile function
	a_k: float
		King amplitude
	beam_fwhm: float
		fwhm of beam to convolve by (and of point source), in units of arcminutes
	metric: astropy.cosmology.LambdaCDM, optional
		LambdaCDM model to pass for calculating distances.  Default == LambdaCDM(H0=68, Om0=0.31, Ode0=0.69, Ob0=0.049, Tcmb0=2.725)
	dist: str, optional
		Distance calculation method, "comoving" or "proper" distance.  Default == "comoving"
	slope_like_powerlaw: bool, optional
		If slope parameter should be defined in similar manner to powerlaw in astronomy.  Default = True (see opening note for other way)

	Returns
	-------
	numpy.ndarray, of integrated values over profile
	"""
	sig = beam_fwhm / np.sqrt(8.*np.log(2))
	if dist.lower() == "comoving":
		D_arcmin = metric.comoving_transverse_distance(z).value * np.pi / 180 / 60
	elif dist.lower() == "proper":
		D_arcmin = metric.angular_diameter_distance(z).value * np.pi / 180 / 60
	else:
		print("dist not recognized... assuming you meant comoving...")
		D_arcmin = metric.transverse_comoving_distance(z).value * np.pi / 180 / 60
	if slope_like_powerlaw:	###NOTE: i0e has a constant exp(-np.abs(x)) in front, hence why we have np.exp((-(theta-t)**2)/2/sig**2)
		king_gauss = lambda  t: spsp.gamma(1/2) * spsp.gamma((slope-1)/2) / spsp.gamma(slope/2) * (1+(D_arcmin*t/r_0)**2)**((1-slope)/2) / sig**(2) * t * np.exp((-(thetas-t)**2)/2/sig**2) * spsp.i0e(thetas*t/sig**2)
	else:
		king_gauss = lambda  t: spsp.gamma(1/2) * spsp.gamma((3*slope-1)/2) / spsp.gamma(3*slope/2) * (1+(D_arcmin*t/r_0)**2)**((1-3*slope)/2) / sig**(2) * t * np.exp((-(thetas-t)**2)/2/sig**2) * spsp.i0e(thetas*t/sig**2)
	return a_k * np.asarray(spint.quad_vec(king_gauss, 0, np.inf,workers=1)[0])

def proj_ptsrc_king_convolved(
		thetas, z, r_0,slope, a_k, a_ps, beam_fwhm, c=0, metric=LambdaCDM(H0=68,Om0=0.31,Ode0=0.69,Ob0=0.049,Tcmb0=2.725),
		dist="comoving", beam_lmax=None, slope_like_powerlaw=True):
	"""Point source plus King profile model, gaussian convolved
	
	Parameters
	---------
	thetas: list or array
		Either list or array of thetas to sample profile at.  In units of arcminutes
	z: float
		redshift of profile/sample
	r_0: float
		"Core" or reference radius to scale model by, in Mpc
	slope: float
		slope exponent in profile function
	a_k: float
		King amplitude
	a_ps: float
		Point source amplitude
	beam_fwhm: float
		fwhm of beam to convolve by (and of point source), in units of arcminutes
	c: float, optional
		Constant term/offset (mainly for testing purposes) Default == 0
	metric: astropy.cosmology.LambdaCDM, optional
		LambdaCDM model to pass for calculating distances.  Default == LambdaCDM(H0=68, Om0=0.31, Ode0=0.69, Ob0=0.049, Tcmb0=2.725)
	dist: str, optional
		Distance calculation method, "comoving" or "proper" distance.  Default == "comoving"
	beam_lmax: int, optional
		If a lmax (max ell, spherical harmonic) cutoff for beam.  Used in point source creation (gaussian if None, otherwise uses healpy to make) Default == None
	slope_like_powerlaw: bool, optional
		If slope parameter should be defined in similar manner to powerlaw in astronomy.  Default = True (see opening note for other way)

	Returns
	-------
	numpy.ndarray, of integrated values over profile
	"""
	return proj_ptsrc_convolved(thetas, a_ps, beam_fwhm, beam_lmax) + proj_king_convolved(thetas, z, r_0, slope, a_k, beam_fwhm, metric=metric, dist=dist, slope_like_powerlaw=slope_like_powerlaw) + c

def proj_king(rs, r_0, slope, a_k, slope_like_powerlaw=True):
	"""Project King profile (no gaussian/beam convolution)
	slope_like_powerLaw=True makes the unprojected power of -gamma/2 (matching power law of -gamma), 
	instead of -3*gamma/2 (other standard convention resulting in gamma=1 for projected power equal to 1)
	
	Parameters
	---------
	rs: list or array
		Either list or array of radial positions to sample profile at.  In same units as r_0
	r_0: float
		"Core" or reference radius to scale model by, in same units as rs
	slope: float
		slope exponent in profile function
	a_k: float
		King amplitude
	slope_like_powerlaw: bool, optional
		If slope parameter should be defined in similar manner to powerlaw in astronomy.  Default = True (see opening note for other way)

	Returns
	-------
	numpy.ndarray, of profile values
	"""
	if slope_like_powerlaw:
		return a_k * spsp.gamma(1/2) * spsp.gamma((slope-1)/2) / spsp.gamma(slope/2) * (1+(rs/r_0)**2)**((1-slope)/2)
	else:
		return a_k * spsp.gamma(1/2) * spsp.gamma((3*slope-1)/2) / spsp.gamma(3*slope/2) * (1+(rs/r_0)**2)**((1-3*slope)/2)

def proj_powerlaw(rs, r_0, slope, a_pl):
	"""Projected powerlaw function (no gaussian/beam convolution, especially since power law diverges at zero)

	Parameters
	---------
	rs: list or array
		Either list or array of radial locations to sample profile at.  In same units as r_0
	r_0: float
		"Core" or reference radius to scale model by, in same units as rs
	slope: float
		slope exponent in profile function
	a_pl: float
		Powerlaw amplitude

	Returns
	-------
	numpy.ndarray, of profile values
	"""
	return a_pl * spsp.gamma(1/2) * spsp.gamma((slope-1)/2) / spsp.gamma(slope/2) * (rs/r_0)**(1-slope)


###Some smaller stuff for mean subtracting and covariances scaling
def rad_avg_area(r0,r1):
	"""Returns area within radial average bounds r0 and r1"""
	return (r1**2-r0**2)*np.pi

def sub_outer_bins(array: np.ndarray, rads: np.ndarray, N: int = 3, return_mean_shift: bool = False) -> np.ndarray:
	"""To normalize offset of furthest N bins (radial), return_mean_shift also returns value subtracted from each element
	
	Parameters
	----------
	array: np.ndarray
		>= 2d array (or list or pd.DataFrame) where final dimension corresponds to radial bins
	rads: np.ndarray
		1d array or list corresponding to radial bin edges.  Should technically work if it contains the N+1 furthest edges desired to be subtracted
	N: int, optional
		Default == 3, number of (furthest) radial bins to subtract average from array
	return_mean_shift: bool, optional
		Default == False.  If True, also returns what said mean offset/shift from furthest bins was. Can also be calculated by taking input - output array
	
	Returns
	-------
	np.ndarray, and float optionally (if return_mean_shift == True)
	"""
	m_shift = np.sum(np.array(array)[...,-N:] * rad_avg_area(rads[-(N+1):-1], rads[-N:]), axis=-1) / rad_avg_area(rads[-(N+1)], rads[-1])
	# print(mshift)
	if return_mean_shift:
		return np.array(array) - m_shift[..., None], m_shift
	else:
		return np.array(array) - m_shift[..., None]

def scale_covs(covs: np.ndarray, N: int = 10) -> np.ndarray:
	"""Where covs is bxfxf array of b radial bins, N = # of bins at end that are re-scaled...	i.e. Takes the N+1 furthest bin covariance 
	and uses it for all further N bins (when suspecting the N further bins may have incorrect/underestimated covariances)"""
	if covs.shape[-2] != covs.shape[-1]:
		print("Array doesn't look like bxfxf set of b cov matrices")
		return None
	fs = covs.shape[-1]
	for f1 in range(fs):
		for f2 in range(fs):
			covs[-N:, f1, f2]=covs[-N:, f1, f2] * np.sqrt(covs[-(N+1), f1, f1]) * np.sqrt(covs[-(N+1), f2, f2]) / np.sqrt(covs[-N:, f1, f1]) / np.sqrt(covs[-N:, f2, f2])
	return covs


###Subsequent functions for energy-mass or mass-mass correction function
def x_y_pl(x, x_peak, pl_amplitude, pl_slope):
	""" linear-linear powerlaw function"""
	return pl_amplitude * (x/x_peak)**pl_slope

def xlog_y_pl(xlog, xlog_peak, pl_amplitude, pl_slope, return_linear=True):
	""" log-linear powerlaw function"""
	if return_linear:
		return 10**(np.log10(pl_amplitude) + (xlog-xlog_peak)*pl_slope)
	else:
		return np.log10(pl_amplitude) + (xlog-xlog_peak)*pl_slope

def xlog_ylog_pl(xlog, xlog_peak, log_pl_amplitude, pl_slope):
	""" log-log powerlaw function"""
	return log_pl_amplitude + (xlog-xlog_peak)*pl_slope

def powerlaw_mass_correction_by_convolving(
		logm_range: list or np.ndarray, pl_amplitude: float, pl_slope: float, sed_sigma: float,
		mass_distr_sigma: float, mass_distr_mean: float, pl_lin: bool = True) -> np.ndarray:
	"""Quick way when mass_distr_means = single mean (i.e. mass distr. is Gaussian). 
	Use this when plotting a 1d or 2d range of values needing to be convolved (i.e. plotting a forward-modeled relation)
	
	Parameters
	----------
	logm_range: list or np.ndarray
		set of (log) masses to compute values at
	pl_amplitude: float
		Powerlaw amplitude
	pl_slope: float
		Powerlaw slope
	sed_sigma: float
		Sigma or gaussian uncertainty of SED fitting (we had 0.16 dex uncertainty)
	mass_distr_sigma: float
		Gaussian sigma of (log) mass distribution fit (ours was 0.20 dex)
	mass_distr_mean: float
		Gaussian peak or mean of (log) mass distribution (this ends up also being the median of linear mass)
	pl_lin: bool, optional
		Default == True for when powerlaw is defined in log-linear space (False for log-log)

	Returns
	-------
	np.ndarray (same size as logm_range)
	"""
	
	def real_sigma(old_sigma, factored_sigma):
		return np.sqrt(old_sigma**2 - factored_sigma**2)
	
	def gaussian(x, amp, c, sig):
		return amp / (sig*np.sqrt(2*np.pi)) * np.exp(-((x-c)**2)/2/sig**2)
	
	uncon_sigma=real_sigma(mass_distr_sigma, sed_sigma)
	if pl_lin:	###If y/ returned values are linear space, not log
		def pl_w(log_mass):
			return xlog_y_pl(log_mass, xlog_peak=mass_distr_mean, pl_amplitude=pl_amplitude, pl_slope=pl_slope, return_linear=False) * gaussian(log_mass, 1, mass_distr_mean, uncon_sigma)
	else:
		def pl_w(log_mass):
			return xlog_ylog_pl(log_mass, xlog_peak=mass_distr_mean, log_pl_amplitude=pl_amplitude, pl_slope=pl_slope) * gaussian(log_mass, 1, mass_distr_mean, uncon_sigma)

	if len(logm_range.shape) > 1:	###Checking if multidimensional and then second axis is longer than length of 1 (which will be raveled...)
		if logm_range.shape[1] > 1:
			print("Cannot pass multidimensional m_range through this function due to convolution... ending before an error or incorrect value is returned", time.asctime())
			exit()
		g1 = gaussian(logm_range.ravel(), 1, np.mean(logm_range), sed_sigma)
		g2 = gaussian(logm_range.ravel(), 1, mass_distr_mean, uncon_sigma)
		return spi.convolve1d(pl_w(logm_range), g1,mode="constant", origin=0, axis=0) / spi.convolve1d(g1, g2, mode="constant", origin=0)[...,None]
	else:	###i.e. m_range is a 1d array, which is fine
		g1 = gaussian(logm_range, 1, np.mean(logm_range), sed_sigma)
		g2 = gaussian(logm_range, 1, mass_distr_mean, uncon_sigma)
		return spi.convolve1d(pl_w(logm_range), g1, mode="constant", origin=0, axis=-1) / spi.convolve1d(g1, g2, mode="constant", origin=0)

def powerlaw_mass_correction_mbinned(
		mbin_centers: list or np.ndarray, mbin_widths: float or list or np.ndarray, pl_amplitude: float, pl_slope: float, sed_sigma: float,
		mass_distr_sigmas: float or list or np.ndarray, mass_distr_means: float or list or np.ndarray, norms: list or np.ndarray=None, pl_lin: bool=True) -> np.ndarray:
	"""Mass correction of a given set of mass bin measurements (given in log10(M/M_sun) dex), 
	where the mass distribution is described by a set of Gaussians and said masses have their own sed_sigma and 
	the expected unconvolved function is a power-law: A*(M/M_p)^alpha.  M_p is the peak mass as determined by the provided gaussian fit of the original mass distribution.
	**Notes: sed_sigma must be >0.08 else may cause a RuntimeWarning with scipy.integrate.quad_vec and scipy.special.erfc()

	
	Parameters
	--------------

	mbin_centers: list or np.ndarray
		1d (log) mass bin centers of bin values.
	mbin_widths: float or list or np.ndarray
		Float for equal (log) widths, or 1d list/np.ndarray same size as mbin_centers
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
	pl_lin: bool, optional
		Default == True, If "pl_peak" is sought in it's linear form (i.e. for fitting, equal steps in linear), else False = log form instead

	Returns
	-------
	np.ndarray
	
	"""

	def real_sigma(old_sigma,factored_sigma):
		return np.sqrt(old_sigma**2-factored_sigma**2)
	
	def Gaussian(x, amp, c, sig):
		return amp / (sig*np.sqrt(2*np.pi)) * np.exp(-((x-c)**2)/2/sig**2)

	def Gint(log_mass, sigma,lower,upper):
		return (spsp.erf((upper-log_mass) / sigma / np.sqrt(2)) - spsp.erf((lower-log_mass) / sigma / np.sqrt(2))) / 2

	def Gint2(log_mass, sigma, lower, upper):	###New, slightly better way to not cause the weird runTime error quite as fast...
		return spint.quad_vec(lambda x: Gaussian(x,1,log_mass,sigma),lower,upper)[0]

	if norms is not None:
		print("Mass distribution not an individual Gaussian...")
		norms = np.array(norms) / np.sum(norms)			###Normalizing the norms
		uncon_sigmas = real_sigma(np.array(mass_distr_sigmas), sed_sigma)
		m_check = np.arange(0, 20*(np.max(mbin_centers) + np.max(mbin_widths)), 0.01)
		# mass_peak = 10**m_check[np.argmax(np.sum(Gaussian(m_check[...,None], norms, np.array(mass_distr_means), np.array(mass_distr_sigmas)), axis=1))]
		logmass_peak = m_check[np.argmax(np.sum(Gaussian(m_check[...,None], norms, np.array(mass_distr_means), np.array(mass_distr_sigmas)), axis=1))]	###For log method (to validate that I get the same answer as old way)
		def pl_weight(log_mass, m_lower, m_upper):
			return np.sum(Gaussian(log_mass, norms, np.array(mass_distr_means), np.array(uncon_sigmas))) * Gint2(log_mass, sed_sigma, m_lower, m_upper)
	else:
		uncon_sigma = real_sigma(mass_distr_sigmas, sed_sigma)
		logmass_peak = mass_distr_means	###For log method (to validate that I get the same answer as old way)
		def pl_weight(log_mass, m_lower, m_upper):
			return Gaussian(log_mass, 1, mass_distr_means, uncon_sigma) * Gint2(log_mass, sed_sigma, m_lower, m_upper)
	if pl_lin:
		def pl(log_mass, m_lower, m_upper):
			###Log-log method (bc fitting procedure if given is_linear=True plugs in zero...)
			return ((xlog_y_pl(log_mass, xlog_peak=logmass_peak, pl_amplitude=pl_amplitude, pl_slope=pl_slope))) * pl_weight(log_mass, m_lower, m_upper)	
	else:
		def pl(log_mass, m_lower, m_upper):
			return xlog_ylog_pl(log_mass, xlog_peak=logmass_peak, log_pl_amplitude=pl_amplitude, pl_slope=pl_slope) * pl_weight(log_mass, m_lower, m_upper)	
	integral_min = 1
	###Setting constant since doesn't matter too much... and we don't expect super massive galaxies...
	integral_max = 20 #np.ceil(2*np.max(mass_distr_means)+5*(np.max(mass_distr_sigmas)+sed_sigma))

	if isinstance(mbin_widths, (list, np.ndarray)) and len(mbin_widths) == len(mbin_centers): ###if a width is given per bin
		return 10**np.array([spint.quad_vec(lambda x: pl(x, mbin_centers[m] - mbin_widths[m]/2, mbin_centers[m] + mbin_widths[m]/2), integral_min, integral_max)[0] / spint.quad_vec(lambda x: pl_weight(x, mbin_centers[m] - mbin_widths[m]/2,mbin_centers[m] + mbin_widths[m]/2), integral_min, integral_max)[0] for m in range(len(mbin_centers))])
	elif isinstance(mbin_widths, (int, float)):
		return 10**np.array([spint.quad_vec(lambda x: pl(x, mbin_centers[m] - mbin_widths/2, mbin_centers[m] + mbin_widths/2), integral_min, integral_max)[0] / spint.quad_vec(lambda x: pl_weight(x, mbin_centers[m] - mbin_widths/2, mbin_centers[m] + mbin_widths/2), integral_min, integral_max)[0] for m in range(len(mbin_centers))])
	else:	###Provided incorrect data, I think should throw and error somewhere before
		print("Check dtype of mbin_widths...")
		return None


if __name__ == "__main__":    ###For any other quick testing purposes
	print("Start Time: ", time.asctime())


	
	print("End Time: ", time.asctime())