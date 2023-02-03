from __future__ import division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import scipy.special as spsp
import warnings
import gc
from typing import Any, Callable, List

##################General Fitting##################
####My new generalized fitting procedure....
class FuncParam:
	def __init__(
			self, func_name: str = "", text_name: str = r"", fit_range: float or np.ndarray = np.arange(0, 1.01, 0.1), 
			prior_weight: np.ndarray = None, is_linear: bool = False):
		"""Function parameter class object with the below attributes, 
		to be used in function fitting processes of other func_fitting.py functions.

		Attributes
		----------
		func_name: str (Default = "")
			What the function's parameter is called (kwarg) to pass through to the function.
		
		text_name: str (Default = r"")
			The parameter's official name once printed on a plot or in-text (i.e. LaTeX format).
			This will be used for any plots, tables, etc.
		
		fit_range: int, float, or array (Default = np.arange(0,1.01,0.1) == array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
			Range of desired values to fit for (When requiring probability density calculations such as BE).
			If int, float, or array with len==1, it will be considered a constant and simply passed onto the fitting function as such (not fit for)...
			For GLS, fit_range only dictates if it should be considered a constant (if int or float) or free-fit (if array with len > 1), 
			since fit can be determined exactly without generating the probability density.
		
		prior_weight: None or array  (Default = None)
			Use if fitting is wanted with any priors applied.  I.e. weighting certain fit values more than others.
			Should be same size as fit_range. Ignored if fit_range is a constant.

		is_linear: bool (Default = False)
			If you wish to identify when a parameter is linear, simplifying BE fitting.  Ignored in GLS fitting (already assumed linear)

		"""

		self.func_name = func_name
		self.text_name = text_name
		self.fit_range = fit_range
		self.prior_weight = prior_weight
		self.has_prior = False
		self.is_linear = is_linear
		if isinstance(fit_range, int) or isinstance(fit_range, float):
			self.constant = True
		elif len(fit_range) == 1:
			warnings.warn("%s: Length of fit_range is 1, assuming it is a constant and changing it to a float..."%func_name)
			self.fit_range = float(fit_range[0])
			self.constant = True
		else:
			self.constant = False
			if prior_weight is not None:
				if fit_range.shape != prior_weight.shape:
					raise ValueError("%s: A prior weight was assigned but is not the same shape as the provided fit_range..."%func_name)
				else:
					self.has_prior = True
	pass


def prob_level(prob_matrix, percentile):
	"""Given a probability density matrix, returns the level containing the desired percentile ABOVE that level (0 < percentile < 1).  
	I.e. All probabilities >= returned level correspond to that interval.
	This assumes the probability matrix is considered roughly continuous and similar to a normal distribution (only one definite peak).
	
	Parameters
	----------
	prob_matrix: np.ndarray
		multi-dimensional numpy array probability matrix to find levels of
	percentile: float
		Float between 0 and 1 of desired percentile level to find

	Returns
	-------
	float: Level (between 0 and 1) corresponding to desired percentile
	"""
	
	flattened = prob_matrix.flatten()
	flattened.sort()
	csum = np.cumsum(flattened) / np.sum(flattened)
	return np.min(flattened[csum >= (1-percentile)])

def ci_edges(prob_1d, prob_range, ci):
	"""Given a 1D probability density, returns bounds [lower, upper] of **2-tail** Confidence/Credible Interval containing the CI percentile amount.
		For best results, probabilities should go to zero at lowest and highest provided range value

	Parameters
	----------
	prob_1d: list or np.ndarray
		1d list or array of probability density to find bounds of
	prob_range: list or np.ndarray
		1d range of values said 1d probability density corresponds to
	ci: float
		confidence interval (from 0 to 1) to seek edges/bounds of. I.e. .8 corresponds to bounds containing 80% of probability distribution
	
	Returns
	-------
	list (2 elements): 
	lower and upper prob_range values containing desired confidence/credible interval
	"""
	csum=np.cumsum(prob_1d) / np.sum(prob_1d)
	return [np.min(prob_range[csum >= (1-ci)/2]), np.min(prob_range[csum >= (1+ci)/2])]		###has to be min bc arrays must be of size>0

def gaussian_range_prior(mean, sigma, step_resolution, sigma_range=3):
	"""returns a fit_range and fit_prior_weight about a desired mean/center, and out to 3 sigma (Default sigma_range=3)
	For use with FuncParam class to pass through func_fit()
	
	Parameters
	----------
	mean: float
		Gaussian mean
	sigma: float
		Gaussian sigma
	step_resolution: float
		desired resolution of range
	sigma_range: float, optional
		Sigma level to extend range out to (from provided mean).  Default == 3, i.e. ~99.7%
	
	Returns
	-------
	1d np.ndarray, 1d np.ndarray: 
	fit_range and prior_weights meant to be passed to FuncParam
	"""
	
	half=np.arange(0, sigma_range*sigma, step_resolution)
	fit_range=np.concatenate([-half[:0:-1], half]) + mean
	prior_weight = np.exp(-((fit_range-mean) / sigma)**2/2) / np.sqrt(2*np.pi) / sigma
	return fit_range, prior_weight

def sigma_levels_nd_gaussian(n, s):
	"""Calculates percentage (0. to 1.) of an n-dimensional (radially-symmetric) gaussian contained within s-sigma
	Such as sigma_levels_nd_Gaussian(1,2) -> 0.9544997361036415
	
	Parameters
	----------
	n: int
		Dimension of Gaussian to calculate
	s: float
		Sigma level to calculate within on Gaussian

	Returns
	-------
	float: Percentage (from 0 to 1) of Gaussian contained within s-sigma
	"""
	return spsp.gammainc(n/2, s**(2)/2)

def func_fit(
	func_to_fit: Callable, x_values: np.ndarray or list, y_values: np.ndarray, func_params: List["FuncParam"], variance: float or np.ndarray = None, method: str = "BE", 
	best_fit: str = "median", absolute_variance: bool = True, MemoryOverride: bool = False, ci_method: str = "standard", showBnds_1D: list or np.ndarray = [1], showFitValues: bool = True, 
	showBnds_2D: list or np.ndarray = [1,2,3], quick_corner_plot: bool = True, corner_plot_save_name: str = None, save_format: str = "png", within_n_dim_bnds: list = [], return_marg_dists: list = None, font_size: float = 12, verbose: bool = False, **func_to_fit_kwargs):
	"""Using BE (Bayesian Estimation, Default) or GLS (Generalized Least Squares) to fit given data to the desired func_to_fit function.
	
	Parameters
	----------
	func_to_fit: Callable 
		function(x_values, func_params)
		Desired function to fit. Should be capable of numpy array scaling/vectorizing.
		Requires a set of independent x_values and func_params parameter objects to pass through func_toFit.  
		
	
	x_values: list, array, or list of arrays (<- emphasis on LIST of arrays)
		Independent variable(s). 
		Will be converted to np.ndarray by stacking along the axis=-1 to preserve numpy vectorization??...
	
	y_values: list or array
		Dependent variables(s)/measurements.  Should be same lenth as x_values. 
		Will be converted to np.ndarray with np.asarray()...
	
	func_params: list of FuncParam
		List of func_param(func_name, text_name, fit_range, and prior_weight) objects
		func_name == kwarg name to be passed through func_toFit
		text_name == raw string name to be pass through all plots, etc.  i.e. r"2 S$_\nu$"
		fit_range == set of all fit values to attempt to fit for.  If constant, provide as int, float, or len==1 array (or pass as **func_to_fit_kwargs).  
		Array only used in generating BE probability density.  GLS only looks at if non-constant (array of len>1)
		prior_weight == If any prior is desired, must be same size as fit_range.  Fit with method="GLS" does not currently take any priors into account.
		Any func_param assigned a fit_range of int, float, or 1D array with len==1, will be interpreted as a desired constant (not fit for)
		For method "BE" (Default), each fitting parameter's fit_range must be assigned a 1D array (len > 1) covering the desired values to generate the probabilty density over.
		For method "GLS", each fitting parameter's fit_range must be assigned a 1D array (len > 1) to be considered non-constant.  
		If non-constant (np.ndarray w/ len > 1), actual fit_range values do not matter (GLS does not generate probability densities)
	
	variance: None, float, 1D array, or 2D array (covariance matrix)
		(Default) None == All variances are assumed equal-weighted.
		Float == All variances assumed same provided float value.  
		1D array == Individual variances for each measurement, assumed independent from each other.
		2D array (Covariance Matrix) == Full covariance matrix, equivalent to 1D if off-diagonals are all zero (further equivalent to float if all same).

	method: str, optional
		Either "BE" or "GLS"
		(Default) "BE" == Bayesian Estimation.  Valid for non-linear fits (careful defining ranges for func_toFitParams which can greatly impact fit & accuracy)
		"GLS" == Generalized Least Squares.  Valid for linear fits (Regression), often much faster since probability densities do not have to be generated.  Priors are not considered.
	
	best_fit: str, optional
		Either "median", "average", "peak1d", or "peak"
		(Default) "median" == Best fit reported is median of 1D parameter densities (50th percentile).  Commonly used in astronomy & academic results.  Median will also be reported if unrecognized value is passed.
		"average" == Best fit reported is mean of 1D parameter density distribution.
		"peak1d" == Highest probability of 1D distribution.
		"peak" == Highest probability of p-dimensional distribution (for p total parameters fit).

	absolute_variance: bool, optional
		(Default) True == Variances passed represents the variance in absolute sense
		False == Variances passed are only relative magnitudes.  Final reported covariance matrix is scaled by the sample variance of the residuals of the fit
	
	MemoryOverride: bool, optional 
		(Default == False) If you want to bypass a warning inplace to prevent BE from creating large arrays over ~4 GB in size.

	ci_method: str, optional
		Confidence Interval method for 1D bounds.  (Default) "standard" == quantile selection where lower and upper bounds exlude an equal percentage. 
		Else, calculates most probable marginalized densities first (not recommended, but was old way).

	showBnds_1D: list or ndarray = [1] : 
		(Default) [1] (show only 1 sigma bounds)
		Can select [1,2,3] corresponding to 1, 2, or 3 sigma 1D bounds.
	
	showFitValues: bool = True : 
	Show best fit values and associated (1-sigma) errors above diagonals of corner plot.  1-sigma errors are currently only option to show as of now.

	showBnds_2D: list or ndarray = [1,2,3] : 
		(Default) [1,2,3] corresponding to 1, 2, or 3 sigma, 2D bounds.
		Show sigma bounds on 2D contour, can select any set of sigma bounds, but need to be monatonically increasing. 

	quick_corner_plot: bool, optional
		(Default) True == Generate and show corner plot of parameter posterior distributions (For method="BE" only)

	corner_plot_save_name: str, optional
		If quick_corner_plot == True. (Default) "", results in corner plot showing and not saving.  Any other string will result in it being saved.
	
	within_n_dim_bnds: list = [] : 
		List of sigma bounds to return all possible (n #) parameter combinations within.  Corresponds to n-dimensional gaussian bounds, used in func_fitPlot.
	
	return_marg_dists: list = None
		list (or list of lists) of desired densities to return marginal distributions of.  
		Example: For func_params (to fit) "a","b", and "c" , a return_marg_densities=[["a"],["a","b"]] will return a dictionary of:
		A 1-d prob density array of marginal distribution of "a" (marginalized over "b" and "c"), 
		and a 2-d prob density array of marginal distribution of "a" and "b", (marginalized over "c")

	**func_to_fit_kwargs: 
		Any additional kwargs needed passed through func_toFit.

	"""
	if verbose:
		print("---- func_fit of %s ----"%str(func_to_fit.__name__))
	
	###SETUP
	warnings.filterwarnings("error", category=RuntimeWarning)
	## Just in case formatting
	x_values = np.stack(x_values, axis=-1)		## We want to ensure vectorization is preserved by stacking along the last dimension
	y_values = np.asarray(y_values)
	N = len(y_values)
	## Sorting parameters into constants and non-constants
	const_func_params = [p for p in func_params if p.constant]
	const_func_kwargs = dict([[p.func_name,p.fit_range] for p in const_func_params])	# dictionary of constant kwargs to pass through function
	tofit_func_params = [p for p in func_params if not p.constant]
	fit_param_len = len(tofit_func_params)
	fit_param_shape = [len(p.fit_range) for p in tofit_func_params]

	if verbose:
		print("y_values length: ", N)
		print("Constant Parameters: ", const_func_kwargs)
		print("Variable/to-Fit Parameter Function Names: ", [p.func_name for p in tofit_func_params])

	## Inverse covariance matrix
	if variance is None:							## Then assume all y values are evenly weighted
		inv_cov = np.identity(N)
	elif np.prod(np.asarray(variance).shape) == 1:	## i.e. given one error assumed for all (with zero correlation/off-diagonals)
		inv_cov = np.identity(N) / variance
	elif len(np.asarray(variance).shape) == 1:		## i.e. given y errors, assumes off-diagonals are zero (i.e. no covariance between y measurements)
		if len(np.asarray(variance)) != N:
			raise ValueError("1D variance given but is not the same size as provided x-values...")
		inv_cov = np.identity(N)
		for i in range(N):
			inv_cov[i,i] = variance[i]**(-1)		###Filling diagonals
	else:		## Assuming now a full covariance matrix was given
		inv_cov = np.linalg.inv(variance)
	
	###Let's do the GLS version first:
	if method.upper() == "GLS":
		print("-----GLS Fitting-----")
		## Gotta generate the linear X matrix terms (what the parameters are multipled by)
		## Create a dictionary of kwargs from identity matrix for parameter factors
		tofit_iden = np.identity(fit_param_len)
		tofit_iden_func_kwargs = dict([[tofit_func_params[p].func_name, tofit_iden[p][None,...]] for p in range(fit_param_len)])	#dictionary of constant kwargs to pass through function
		if verbose:
			print("Constants: \n", tofit_iden_func_kwargs)
		x_matrix = func_to_fit(x_values[...,None], **const_func_kwargs, **tofit_iden_func_kwargs, **func_to_fit_kwargs)	###X-matrix, shape of (y_values len,fit_func_params len), passing all constants as given in case any were nonlinear
		const_offset = func_to_fit(x_values, **const_func_kwargs, **dict([[p.func_name, 0] for p in tofit_func_params]), **func_to_fit_kwargs)
		## In case any of the constants were nonlinear, we now subtract our constant term from the X matrix (removing the columns of 1s that would otherwise exist in the X matrix)
		x_matrix = x_matrix - np.expand_dims(const_offset, 1)		
		## Similarly, we subtract the constant term from the measured y_vector.  Any constant term will have now been accurately removed from the array.  If fitting of it was instead desired, designate it in the func_toFit and as a fit_param
		y_vector = (y_values - const_offset).T

		## The actual fit		
		best_fit_params = np.linalg.inv(x_matrix.T @inv_cov@x_matrix) @ x_matrix.T @ inv_cov @ y_vector
		if verbose:
			print("GLS Best Fit: \n", best_fit_params)
		scale=1
		if not absolute_variance:
			scale=(y_vector - x_matrix@best_fit_params).T @ (y_vector - x_matrix@best_fit_params) / (N-len(best_fit_params))
			if verbose:
				print("Variance scaled by: ",scale)
		
		fit_param_cov = np.linalg.inv(x_matrix.T@inv_cov@x_matrix) * scale


	## Now the BE method:
	elif method.upper() == "BE":
		print("-----BE Fitting-----")
		fit_ranges = [p.fit_range for p in tofit_func_params]	###All ranges to fit
		nonlinear_tofit_func_params = [p for p in tofit_func_params if not p.is_linear]		###All non-linear parameters to fit
		nonlinear_fit_ranges = [p.fit_range for p in nonlinear_tofit_func_params]				###All non-linear ranges to fit
		nonlinear_len = len(nonlinear_fit_ranges)
		linear_tofit_func_params = [p for p in tofit_func_params if p.is_linear]				###All linear parameters to fit
		linear_fit_ranges = [p.fit_range for p in linear_tofit_func_params]					###All linear ranges to fit
		linear_ref = np.asarray([p for p in range(fit_param_len) if tofit_func_params[p].is_linear])		###For creating dimensions of grid array
		prior_check = [p.has_prior for p in tofit_func_params]
		## Checking array size needed
		max_param_size = np.max([fit_param_len, N]) * np.prod([len(r) for r in fit_ranges])
		if verbose:
			print("Estimated Max parameter array size product: ", max_param_size)		###Check
		if max_param_size*64/8 > (4*10**9) and not MemoryOverride:
			raise MemoryError("Expected memory use (%.2f GB) from parameter size is greater than 4 GB.... you may want to reduce your parameters' fit_range. Set MemoryOverride=True to override this caution."%(max_param_size*8/10**9))
		## New Way!!! Now first creating grid only of non-linear terms needed to fit with combination of 0 or 1 for linear terms (finding each linear term per combination)
		## This helps reduce time and memory usage, if, random example, someone has a 2d x_values that's 100s of elements long... *COUGH*
		linear_tofit_iden = np.identity(len(linear_fit_ranges))
		if verbose:
			print("Linear Parameter Identity Matrix check: \n", linear_tofit_iden)
		## Get constant offset (when linears are all zero) first
		constant_offset = func_to_fit(np.expand_dims(x_values, axis=tuple(np.arange(nonlinear_len)+len(x_values.shape))), **const_func_kwargs, **dict([[nonlinear_tofit_func_params[p].func_name, np.expand_dims(nonlinear_fit_ranges[p], axis=tuple(np.delete(np.arange(nonlinear_len), p)))] for p in range(nonlinear_len)]),
		**dict([[p.func_name, 0] for p in linear_tofit_func_params]), **func_to_fit_kwargs)	###Holding all linear terms to 0 to find constant independent term
		## Now for each linear term (cycle through each term equaling 1 while the other still zero)
		linear_iden_terms = [func_to_fit(np.expand_dims(x_values, axis=tuple(np.arange(nonlinear_len) + len(x_values.shape))), **const_func_kwargs, **dict([[nonlinear_tofit_func_params[p].func_name, np.expand_dims(nonlinear_fit_ranges[p], axis=tuple(np.delete(np.arange(nonlinear_len), p)))] for p in range(nonlinear_len)]),
		**dict([[linear_tofit_func_params[p].func_name, linear_tofit_iden[i,p]] for p in range(len(linear_tofit_func_params))]),**func_to_fit_kwargs) for i in range(len(linear_fit_ranges))]
		nonlinear_fit_param_grid = None; del nonlinear_fit_param_grid		## To try to save memory? Hopefully? (IDK)
		## Shaping back into grid, expand out axes for linear terms, multiply each component though, add together, flatten back out... so simple!!!
		fit_y_grid = np.expand_dims(constant_offset, tuple(linear_ref+1))
		constant_offset = None; del constant_offset
		gc.collect()								## Maybe helpful to have here?
		fit_y_grid_copy = np.copy(fit_y_grid)		## The constant term to be subtracted from each linear term that was calculated
		for i in range(len(linear_fit_ranges)):
			fit_y_grid = fit_y_grid + (np.expand_dims(linear_iden_terms[i], tuple(linear_ref+1))-fit_y_grid_copy) * np.expand_dims(linear_fit_ranges[i], [0] + list(np.delete(np.arange(fit_param_len)+1, linear_ref[i])))
		
		## Now remove unnecessary variable (memory help?)
		fit_y_grid_copy = None; linear_terms = None; constant_offset = None; linear_iden_terms = None; del linear_terms, constant_offset, linear_iden_terms, fit_y_grid_copy
		fit_y_grid = fit_y_grid.reshape(len(y_values), np.prod(fit_y_grid.shape[1:]))
		residuals = y_values[...,None] - fit_y_grid
		fit_param_grid = None; fit_y_grid = None; del fit_param_grid, fit_y_grid		###Since we shouldn't need these anymore and want to try to save memory
		prob_density = np.exp(-np.einsum("ij,ji->i", residuals.T, inv_cov@residuals) / 2).reshape(fit_param_shape)
		residuals = None; del residuals
		gc.collect()								## Maybe helpful to have here?
		## Apply any priors to our likelihood/probability
		if any(prior_check):		###If any priors were assigned...
			print("At least 1 Prior Weight Provided \n")
			for p in range(fit_param_len):	###cycle through
				if prior_check[p]:			###If a prior...
					prob_density *= np.expand_dims(tofit_func_params[p].prior_weight, tuple(np.delete(np.arange(fit_param_len), p)))	###Expand prior weight out to same dims as prob_density and multiply with

		## Generating 1D densities for variances and corner plot
		fit_param_cov = np.zeros([fit_param_len, fit_param_len])
		fit_param_1d_densities = []
		best_fit_params = []
		fit_param_1d_error_bnds = []
		for p in range(fit_param_len):
			###marginalizing over parameters to form 1D densities
			p_1d_density = np.sum(prob_density, axis=tuple(np.delete(np.arange(fit_param_len), p)))		###Summing along all axes except the parameter of interest
			p_1d_density /= np.sum(p_1d_density)			###Normalized
			fit_param_1d_densities.append(p_1d_density)

			###Calculating confidence interval bounds out to 3 sigma in 1D marginalized probabilities
			if ci_method.lower() == "standard":	###Selecting percentiles upper and lower
				fit_param_1d_error_bnds.append([ci_edges(p_1d_density, fit_ranges[p], sigma_levels_nd_gaussian(1, i+1)) for i in range(3)])	###cycling through 1-3 sigma levels
			else:	###Old way... selecting highest probabilities first til reaching desired interval/level... not standard and doesn't work well with non-normal distributions (when multiple peaks)
				s1 = prob_level(p_1d_density, sigma_levels_nd_gaussian(1, 1))		###1 Sigma level
				s2 = prob_level(p_1d_density, sigma_levels_nd_gaussian(1, 2))		###2-Sigma level
				s3 = prob_level(p_1d_density, sigma_levels_nd_gaussian(1, 3))		###3-Sigma level
				s_levels = [s1, s2, s3]
				errorBnds = [[np.min(fit_ranges[p][p_1d_density >= s]), np.max(fit_ranges[p][p_1d_density >= s])] for s in s_levels]
				fit_param_1d_error_bnds.append(errorBnds)
			
			###Determining the "Best Fit" value of the parameter
			if best_fit.lower() == "peak1d":
				best_fit_params.append(fit_ranges[p][np.argmax(p_1d_density)])
			elif best_fit.lower() == "average":
				best_fit_params.append(np.average(fit_ranges[p], weights=p_1d_density))
			elif best_fit.lower() == "peak":		###Instead determined from master density distribution.... could result in an outlier report as best fit...
					max_loc = np.unravel_index(np.argmax(prob_density), shape=fit_param_shape)
					best_fit_params = [fit_ranges[i][max_loc[i]] for i in range(fit_param_len)]
			else:		###Since throwing an error here would just be a pain, assuming median as the default best_fit choice...
				best_fit_params.append(fit_ranges[p][np.argmin((np.cumsum(p_1d_density)-0.5)**2)])
			

			###Calculating covariance estimate from 1d probability (needed to know what we considered the "Best Fit" first tho...)
			fit_param_cov[p, p] = np.sum(p_1d_density*(fit_ranges[p]-best_fit_params[p])**2)	###a weighted covariances around the determined best fit
		fit_param_1d_error_bnds = np.asarray(fit_param_1d_error_bnds)
		###Generating 2D densities for covariance estimates and corner plot
		fit_param_2d_densities = []
		fit_param_2d_levels = []
		for p1 in range(fit_param_len):
			fit_param_2d_densities.append([])
			fit_param_2d_levels.append([])
			for p2 in range(fit_param_len)[:p1]:
				p_2d_density = np.sum(prob_density, axis=tuple(np.delete(np.arange(fit_param_len), [p1, p2])))		###Summing along all axes except the TWO parameters of interest
				p_2d_density /= np.sum(p_2d_density)		###Normalizing
				fit_param_2d_densities[p1].append(p_2d_density)

				###Determining the levels for 1, 2, and 3 sigma (the correct way now...)
				temp = []
				for s in showBnds_2D:
					temp.append(prob_level(p_2d_density, sigma_levels_nd_gaussian(2, s)))
				for t in range(len(temp[1:])):
					if temp[t+1] >= temp[t]:
						temp[t+1] = np.copy(temp[t]) / 1.000001
				fit_param_2d_levels[p1].append(temp)
				

				###Covariances
				fit_param_cov[p1, p2] = fit_param_cov[p2, p1] = np.sum(p_2d_density * ((fit_ranges[p2][:, None]-best_fit_params[p2]) * (fit_ranges[p1][None, :]-best_fit_params[p1])))
		if verbose:
			print("BE Best Fits: ", best_fit_params)
			print("1D Error Bounds: \n", fit_param_1d_error_bnds, "\n")
			print("Covariance Estimate: \n", fit_param_cov)

		prob_density /= np.max(prob_density)		###Normalize to max
		
		################ Corner Plotting ################
		## Moving this to a separate callable function to reduce the absurd number of parameters passed through func_fit()
		if quick_corner_plot:
			cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["white", "lightblue", "royalblue", "navy"])
			## Create temporary marginalized densities to pass through corner plot code:
	
	###Generating sigma bounds when applicable
	b_bnds = {}
	if  method == "BE" and within_n_dim_bnds:
		for b in within_n_dim_bnds:	## Up to whatever sigma
			s = prob_level(prob_density, sigma_levels_nd_gaussian(fit_param_len, b))		## b-Sigma level (i.e. 1, 2, 3, etc sigma)
			s_locs = np.where(prob_density >= s)
			param_bnds = {tofit_func_params[f].func_name : fit_ranges[f][s_locs[f]] for f in range(fit_param_len)}
			b_bnds[r"%.1f $\sigma$"%b] = param_bnds
	best_fit_params = dict([[tofit_func_params[f].func_name, best_fit_params[f]] for f in range(fit_param_len)])

	marg_dists = {}
	if method == "BE" and bool(return_marg_dists):	###For when densities are also desired
			tofit_func_params_keys = [p.func_name for p in tofit_func_params]
			if isinstance(return_marg_dists[0], list):	###List of lists:
				for margs in return_marg_dists:
					marg_dists["Marginal Distribution of: " + ", ".join(margs)] = np.sum(prob_density, axis=tuple(tofit_func_params_keys.index(k) for k in tofit_func_params_keys if k not in margs))
			else:
				marg_dists["Marginal Distribution of: " + ", ".join(return_marg_dists)] = np.sum(prob_density, axis=tuple(tofit_func_params_keys.index(k) for k in tofit_func_params_keys if k not in return_marg_dists))
			
			# t = True
			# ## If single density desired
			# if np.prod(return_marg_densities.shape) > 0:
			# 	t = type(return_marg_densities[0]) is not list
			# if len(return_marg_densities.shape) == 1 and t:
			# 	# temp = np.sum(prob_density, axis=tuple(tofit_func_params_keys.index(k) for k in return_marg_densities))
			# 	# marg_densities[("Marginalized over: " + ", ".join(list(return_marg_densities)))] = temp / np.sum(temp)
			# else:
			# 	for r in return_marg_densities:
			# 		# temp = np.sum(prob_density, axis=tuple(tofit_func_params_keys.index(k) for k in r))
			# 		# marg_densities[("Marginalized over: " + ", ".join(list(r)))] = temp / np.sum(temp)
	
	gc.collect()	###To try to keep memory usage in check (for loops, etc)
	if method == "GLS":	###Return Best Fit and covariance
		return best_fit_params, fit_param_cov
	elif method == "BE":	###Return Best Fit, covariance, 1d errorbands for each parameter, and 1,2,3-sigma bounds for full x_oversampled
		fit_param_1d_error_bnds = dict([[tofit_func_params[f].func_name, fit_param_1d_error_bnds[f]] for f in range(fit_param_len)])

		return best_fit_params, fit_param_cov, fit_param_1d_error_bnds, b_bnds, marg_dists


###Since func_fit is getting too cumbersome...
def plot_func_fit(func_to_plot : Callable, x_values : list or np.ndarray, y_values : list or np.ndarray, yerrs: list or np.ndarray = None, x_showFit_vals : None or list or np.ndarray = None, val_label : str = "Data", axis_labels : list or np.ndarray = [r"x",r"y"], 
best_fit_params: dict = None, fit_param_combos : dict = None, linear_paramNames : list or np.ndarray = None, param_scale: dict = None, xlims : list or tuple = [], ylims : list or tuple = [], xscale : str = None, yscale : str = None, grid : bool = True, 
cmap : mpl.colors.Colormap = mpl.colors.LinearSegmentedColormap.from_list("",["white","lightblue","royalblue","navy"]), cmap_sigma_max : float = 4, discrete_colors : list or np.ndarray = None, save_name : str = None, save_format : str = "png", continue_plot : bool = False, font_size : float = 12, **func_toplot_kwargs):
	"""Takes b_bnds output from funcFit and plots sigma bounds along with any provided data.
	
	Parameters
	--------------
	func_to_plot: Callable : 
	Function to plot best fit and sigma bounds

	x_values: list or np.ndarray  : 
	1D locations to plot, may be different than x_values that would be passed to func in fitting process

	y_values: list or np.ndarray : 
	1D array, measured values to plot

	yerrs: list or np.ndarray = None : 
	yerrs to be passed directly through to plt.errorbar(), can be 1D list/array or 2D with a row for min and max (respectively)... 
	see matplotlib.pyplot.errorbar for more specifics.

	x_showFit_vals: None or list or np.ndarray : 
	x_values to be used for plotting best fit and sigma bounds, will be passed through func_toPlot.  
	If None, it will be generated from 0.9 to 1.1 times min and max of x_values (respectively) for best fit (and sigma bounds if fit_param_combos is still given).

	val_label: str = "Data" : 
	Label for data points in plot legend.

	axis_labels: list or np.ndarray = [r"x",r"y"] : 
	Labels for plot axes.

	best_fit_params: dict = None : 
	Dictionary of best fit param from funFit output[0].

	fit_param_combos: dict = None : 
	Nested dictionary of fit param combinations that make up sigma bounds, from funcFit output[3].

	linear_paramNames: list or np.ndarray = None : 
	1D list or array to specify any parameters that are linear (such as those defined as is_linear=True in func_param class).  Should be func_name that will be passed to func_toPlot.

	param_scale: dict = None : 
	Dictionary in the event of any parameters need scaling (such as by some factor of 10) to match data points.  Keys should be same func_name as fit params.

	xlims: list or tuple = [] : 
	Plot x-limits to pass through plt.xlims()

	ylims: list or tuple = [] : 
	Plot y-limits to pass through plt.ylims()

	xscale: str = None : 
	String to pass through plt.xscale(), such as "log" or "symlog"
	
	yscale: str = None : 
	String to pass through plt.yscale(), such as "log" or "symlog"

	grid: bool = True : 
	Show plot grid

	cmap: mpl.colors.Colormap = mpl.colors.LinearSegmentedColormap.from_list("",["white","lightblue","royalblue","navy"])  : 
	Colormap to pass for defining different sigma bounds.  Selection is set to: cmap(0.9-(float(list(fit_param_combos.keys()))-1)/(len(fit_param_combos)))

	cmap_sigma_max: float = 4 : 
	cmap selection is linear, thus need to determine max sigma to classify as 0 (default==white) color

	discrete_colors: list or np.ndarray = None : 
	1D list or array of strings, defining discrete colors of sigma bounds, should be at least as long as number of sigma bounds in fit_param_combos.
	Note: Data points (from x_values and y_values) and best_fit will still use the color of cmap(1.0).
	
	save_name: str = None : 
	Desired savename, excluding format type which will is defined separately.  Make sure to provide file path if different directory is desired.

	save_format: str = "png" : 
	Save format for plot, .png is default, other standard plt.savefig options available such as "jpg", "pdf", etc.

	continue_plot: bool = False : 
	If desired to neither savefig nor show plot, but plot something else in addition afterwards.  
	Such as calling this plot_funcFit mupltiple times for different data sets to compare.  *Make sure to change colormap/colors accordingly

	font_size: float = 12 :
	font size for text in plot.  Both axes will have this exact float pt size, legend will be 2 pts smaller.

	**func_toplot_kwargs : 
	Any additional kwargs to pass through to func_toPlot, such as constants, extra kwargs, etc.


	Returns
	--------------

	None

	Either shows or saves the plot according to your choice in "save_name : str = None" above.

	"""

	if best_fit_params:
		print(func_to_plot(x_showFit_vals, **best_fit_params, **func_toplot_kwargs).shape)
		if list(x_showFit_vals):
			plt.plot(x_showFit_vals, func_to_plot(x_showFit_vals, **best_fit_params, **func_toplot_kwargs), label="Best Fit", color=cmap(1.0), marker="")
		else:
			x_showFit_vals = np.linspace(np.min(x_values)*0.9, np.max(x_values)*1.1, num=50)
			plt.plot(x_showFit_vals, func_to_plot(x_showFit_vals, **best_fit_params, **func_toplot_kwargs), label="Best Fit", color=cmap(1.0), marker="")
		if fit_param_combos:
			mins = []
			maxes = []
			for p in fit_param_combos:
				###Check if any parameters need scaling
				if param_scale:
					for i in param_scale:
						fit_param_combos[p][i] *= param_scale[i]
				if linear_paramNames is not None:
					linears = np.asarray([fit_param_combos[p][l] for l in linear_paramNames])
					nonlin = dict(fit_param_combos[p])
					for l in linear_paramNames:
						del nonlin[l]
					###Check to ensure param_combos isn't being affected
					# print(nonlin.keys())
					# print(param_combos[p].keys())
					constant = func_to_plot(x_showFit_vals[...,None], **nonlin, **dict([[l,0] for l in linear_paramNames]), **func_toplot_kwargs)
					linear_vals = [func_to_plot(x_showFit_vals[...,None], **nonlin, **dict([[l,0] for l in linear_paramNames if l!=i]), **dict([[i,1]]), **func_toplot_kwargs) for i in linear_paramNames]
					result = constant
					for l in range(len(linear_vals)):
						result = result + linears[l]*(linear_vals[l]-constant)
				else:
					result = func_to_plot(x_showFit_vals[...,None], **p, **func_toplot_kwargs)
				mins.append(result.min(axis=1))
				maxes.append(result.max(axis=1))
			for n in range(len(mins))[::-1]:
				###Have to plot in reverse to ensure the narrower bands are not covered up...
				if discrete_colors:
					plt.fill_between(x_showFit_vals, maxes[n], mins[n], label=list(fit_param_combos.keys())[n], color=discrete_colors[n], alpha=0.5)
				else:
					plt.fill_between(x_showFit_vals, maxes[n], mins[n], label=list(fit_param_combos.keys())[n], color=cmap(0.95-0.95*(float(list(fit_param_combos.keys())[n][:-9])-1)/(cmap_sigma_max)), alpha=0.5)		###cmap(0.9-(float(list(fit_param_combos.keys())[n][:-9])-1)/(len(fit_param_combos)))

	###Plotting values, moving to last so they always show above sigma fill-betweens...
	if yerrs:
		plt.errorbar(x_values, y_values, yerr=yerrs, label=val_label, color=cmap(1.0))
	else:
		plt.scatter(x_values, y_values, label=val_label, color=cmap(1.0))

	plt.legend(fontsize=(font_size-2))
	if xlims:
		plt.xlim(xlims)
	if ylims:
		plt.ylim(ylims)
	if xscale:
		plt.xscale(xscale)
	if yscale:
		plt.yscale(yscale)
	
	plt.xlabel(axis_labels[0], size=font_size)
	plt.ylabel(axis_labels[1], size=font_size)
	plt.tick_params(labelsize=font_size)
	if grid:
		plt.grid()
	
	if save_name:
		plt.savefig("%s.%s"%(save_name, save_format), dpi=400, bbox_inches="tight")
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


# def cornerplot_from_margs(
# 		marg_func_params, margs, legend_names=None, showBnds_2D=[1,2,3], axis_mins=None, axis_maxs=None,
# 		cmaps=[mpl.colors.LinearSegmentedColormap.from_list("", ["white", "lightblue", "royalblue", "navy"])], cmap_sigma_max=4, alpha=0.8, font_size=12, save_name=None):
# 	"""Create corner plot from Bayesian Estimate marginalized probabilities such as from output of func_fit()
	
# 	marg_func_params: list
# 		List of all FuncParam, as passed through func_fit().  If multiple sets of margs used, this should be a 
# 		list of lists (each index for each func_fit).  Each list should then contain the same set of FuncParam.func_names.
# 		Order of cornerplot dictated by the order of first list of parameters.
# 	margs: dict or List[dict]
# 		Dict of all 2d marg_dists arrays as created by func_fit's return_marg_dists parameter, or
# 		list of dicts (for multiple data sets)
# 	legend_names: list, optional
# 		List of names to use in legend for each set of margs.  Default == None (no legend made)
# 	axis_mins: list, optional
# 		1d list of axis range min limits, corresponding to same order as provided FuncParams.
# 		Default == None, where mins are extracted from lowest provided FuncParam.fit_range (per uniquely named FuncParam)
# 	axis_maxs: list, optional
# 		1d list of axis range max limits, corresponding to same order as provided FuncParams.
# 		Default == None, where maxs are extracted from lowest provided FuncParam.fit_range (per uniquely named FuncParam)

# 	"""
	
# 	space = 0.2
# 	marg_len = len(marg_func_params)
# 	if isinstance(margs, list):

# 	# marg_1d_names=["Marginalized over: "+", ".join(marg_param_names[:i]+marg_param_names[i+1:]) for i in range(marg_len)]
# 	# marg_names = [["Marginalized over: "+", ".join(marg_param_names[n] for n in range(marg_len) if (n!=i and n!=j)) for i in range(marg_len)] for j in range(marg_len)]
# 	marg_names = np.array(marg_names)
# 	print(marg_names)
# 	print(marg_names.shape)
	
# 	fig,axs = plt.subplots(nrows=marg_len, ncols=marg_len, sharex='col', figsize=(1+2*marg_len, 1+2*marg_len))
# 	###Finding the contour bounds
# 	if not isinstance(margs, list):	###i.e. only one set to plot
# 		margs = [margs]
# 	for m in range(len(margs)):
# 		fit_param_2d_levels = []
# 		for p1 in range(marg_len):
# 			fit_param_2d_levels.append([])
# 			for p2 in range(marg_len)[:p1]:
# 				###Determining the levels for 1, 2, and 3 sigma (the correct way now...)
# 				temp = []
# 				print(marg_names[p1, p2])
# 				for s in showBnds_2D:
# 					temp.append(prob_level(margs[m][marg_names[p1, p2]], sigma_levels_nd_gaussian(2, s)))
# 				for t in range(len(temp[1:])):
# 					if temp[t+1] >= temp[t]:
# 						temp[t+1] = np.copy(temp[t]) / 1.000001
# 				fit_param_2d_levels[p1].append(temp)
# 		print(fit_param_2d_levels)

# 		for i in range(marg_len):	###Corner plot rows
# 			for j in range(marg_len):	###Corner plot cols
# 				if j > i and m == 0:
# 					fig.delaxes(axs[i, j])
# 				elif i == j:		###1D marginalized distributions
# 					axs[i, j].set_yticks([])
# 					if i != 0:
# 						axs[i, j].get_yaxis().set_visible(False)
# 					axs[i, j].plot(margs_1d_ranges[i], margs[m][marg_names[i,i]]/np.max(margs[m][marg_names[i,i]]), marker="", linestyle="-")
# 					axs[i, j].set_ylim([0, 1.1])
# 				elif i > j:
# 					if j != 0:	###i.e. not leftmost
# 						axs[i, j].sharey(axs[i, 0])
# 						axs[i, j].tick_params(labelleft=False)	###To not show tick labels on inside plots
# 					axs[i, j].contourf(*np.meshgrid(margs_1d_ranges[j], margs_1d_ranges[i]), margs[m][marg_names[i][j]].T, levels=[fit_param_2d_levels[i][j][::-1][b] for b in range(len(showBnds_2D))]+[1], colors=[cmaps[m](0.95-0.95*(b-1)/(cmap_sigma_max)) for b in showBnds_2D[::-1]], alpha=alpha)
# 				axs[i, j].minorticks_on()
# 		# if showFitValues:		###For 1 sigma values only (since any other would likely be confusing.... maybe add options later)
# 		# 	if np.max(np.abs(fit_ranges[i])) < 0.01:
# 		# 		try:
# 		# 			factor = np.ceil(abs(np.log10(np.abs(best_fit_params[i]))))
# 		# 			# if factor==np.inf:	###Happens when best fit is zero...
# 		# 		except:
# 		# 			try:
# 		# 				factor=np.ceil(abs(np.log10(fit_param_1d_errorBnds[i,0,1]-best_fit_params[i])))		###So instead use factor of highest bound
# 		# 			except:
# 		# 				factor = np.ceil(np.abs(np.log10(np.max(fit_ranges[i]))))
# 		# 				warnings.warn("Best fit %s and 1 Sigma bound are either not clearly defined or overlapping... you might need to readjust your bounds or resolution..."%tofit_func_params[i].text_name,UserWarning)
# 		# 			# if factor==np.inf:	###ANDDDDD if even the upper bound is also zero... just use max of fit ranges.... chances are resolution needs to be improved....
						
# 		# 		# print(factor)	###CHECK
# 		# 		sf = math.ceil(max(0, -np.log10(np.max(fit_ranges[i][1:]-fit_ranges[i][:-1])*10**factor)))		###Sig fig selection for float display
# 		# 		axs[i, j].set_title(tofit_func_params[i].text_name+r"=$%.*f_{-%.*f}^{+%.*f}\times10^{-%i}$"%(sf, best_fit_params[i]*10**factor, sf, (best_fit_params[i]-fit_param_1d_errorBnds[i,0,0])*10**factor, sf, (fit_param_1d_errorBnds[i,0,1]-best_fit_params[i])*10**factor, factor), size=(font_size-2))
# 		# 	elif np.max(np.abs(fit_ranges[i])) > 1000:
# 		# 		factor = np.floor(abs(np.log10(np.abs(best_fit_params[i]))))
# 		# 		# print(factor)	###CHECK
# 		# 		sf = math.ceil(max(0, -np.log10(np.max(fit_ranges[i][1:]-fit_ranges[i][:-1])/10**factor)))		###Sig fig selection for float display
# 		# 		axs[i, j].set_title(tofit_func_params[i].text_name+r"=$%.*f_{-%.*f}^{+%.*f}\times10^{%i}$"%(sf,best_fit_params[i]/10**factor, sf, (best_fit_params[i]-fit_param_1d_errorBnds[i,0,0])/10**factor, sf, (fit_param_1d_errorBnds[i,0,1]-best_fit_params[i])/10**factor, factor), size=(font_size-2))
# 		# 	else:
# 		# 		sf = math.ceil(max(0,-np.log10(np.max(fit_ranges[i][1:]-fit_ranges[i][:-1]))))		###Sig fig selection for float display
# 		# 		axs[i, j].set_title(tofit_func_params[i].text_name+r"=$%.*f_{-%.*f}^{+%.*f}$"%(sf,best_fit_params[i],sf,best_fit_params[i]-fit_param_1d_errorBnds[i,0,0],sf,fit_param_1d_errorBnds[i,0,1]-best_fit_params[i]),size=(font_size-2))
# 	for i in range(marg_len):
# 		if axis_maxs[i] < 0.2:		###Have to do this AFTER the plot is generated in order to grab the scientific notation text
# 			plt.subplots_adjust(wspace=space, hspace=space)	###General default is 0.2, so increasing slightly to prevent any sci notation overlap
# 			axs[i, 0].ticklabel_format(style="sci", scilimits=(0, 0))
# 			axs[-1, i].ticklabel_format(style="sci", scilimits=(0, 0))
# 			axs[i, 0].figure.canvas.draw()
# 			axs[i, 0].yaxis.get_offset_text().set_visible(False)
# 			axs[-1, i].xaxis.get_offset_text().set_visible(False)
# 			axs[i, 0].set_ylabel(marg_param_text_names[i] + "  [" + axs[i, 0].get_yaxis().get_offset_text().get_text() + "]", size=font_size)
# 			axs[-1, i].set_xlabel(marg_param_text_names[i] + "  [" + axs[-1, i].get_xaxis().get_offset_text().get_text() + "]", size=font_size)
# 		elif axis_maxs[i] > 1000:
# 			plt.subplots_adjust(wspace=space, hspace=space)	###General default is 0.2, so increasing slightly to prevent any sci notation overlap
# 			axs[i, 0].ticklabel_format(style="sci", scilimits=(0,0))
# 			axs[-1, i].ticklabel_format(style="sci", scilimits=(0,0))
# 			axs[i, 0].figure.canvas.draw()
# 			axs[i, 0].yaxis.get_offset_text().set_visible(False)
# 			axs[-1, i].xaxis.get_offset_text().set_visible(False)
# 			axs[i, 0].set_ylabel(marg_param_text_names[i] + "  [" + axs[i,0].get_yaxis().get_offset_text().get_text() + "]",size=font_size)
# 			axs[-1, i].set_xlabel(marg_param_text_names[i] + "  [" + axs[-1,i].get_xaxis().get_offset_text().get_text() + "]",size=font_size)
# 		else:
# 			axs[i, 0].set_ylabel(marg_param_text_names[i], size=font_size)
# 			axs[-1, i].set_xlabel(marg_param_text_names[i], size=font_size)
# 	###Setting upper left corner
# 	axs[0, 0].set_ylabel(marg_param_text_names[0], sfunc_toPlotize=font_size, labelpad=16)
# 	###now ensuring left subplots have same y-axes as bottom subplot's x-axes...
# 	for i in range(1, marg_len):
# 		axs[-1, i].set_xlim(axis_mins[i], axis_maxs[i])
# 		axs[i, 0].set_yticks(axs[-1,i].get_xticks())
# 		axs[i, 0].set_ylim(axs[-1,i].get_xlim())	###I guess the order matters here?? lim needed after ticks or else ticks will overwrite???? Cool.
	
# 	###placing the legend above center... or attempting to....
# 	if legend_names:
# 		proxy = [plt.Rectangle((0, 0), 1, 1, fc=cmaps[i](1.0), alpha=alpha) for i in range(len(legend_names))]
# 		fig.legend(proxy, legend_names, loc="center", bbox_to_anchor=axs[0, 1].get_position(), fontsize=font_size)

# 	if save_name:
# 		plt.savefig(save_name, dpi=400, bbox_inches="tight")
# 	else:
# 		plt.tight_layout()
# 		plt.show()
# 	plt.cla()
# 	plt.clf()
# 	plt.close("all")	###To save memory
# 	return None


def cont_bnds_func_fitted(func_fitted: Callable, x_cont_vals: list or np.ndarray, fit_param_combos: dict, linear_paramNames: list or np.ndarray = [],
	**func_fitted_kwargs):
	###Check if any parameters need scaling
	if linear_paramNames:	###This does not actually make it faster right now... need to figure out a way to...
		linears = np.asarray([fit_param_combos[l] for l in linear_paramNames])
		print(linears.shape)
		nonlin_kys = list(fit_param_combos.keys())
		# print(nonlin_kys)
		for l in linear_paramNames:
			nonlin_kys.remove(l)
		print("nonlinear keys: ",nonlin_kys)
		nonlinears = np.asarray([fit_param_combos[nl] for nl in nonlin_kys])
		nonlinears_unique, nonlinears_unique_inverse = np.unique(nonlinears, axis=-1, return_inverse=True)
		print("-----------------")
		if nonlinears.size == 0 or (len(linear_paramNames)+1)*nonlinears_unique.shape[-1] < nonlinears.shape[-1]:			###i.e. number of combos to pass through are worth it (time-wise...)
			print("linear method employed")
			nonlinears_unique = dict([[nonlin_kys[nl],nonlinears_unique[nl]] for nl in range(len(nonlin_kys))])
			###Constant term when linears==0
			constant = func_fitted(x_cont_vals[...,None],**nonlinears_unique,**dict([[l,0] for l in linear_paramNames]),**func_fitted_kwargs)
			###Now each linear term (when all other linears are still 0), subtracting constant term
			linear_terms = [(func_fitted(x_cont_vals[...,None],**nonlinears_unique,**dict([[l,0] for l in linear_paramNames if l!=i]),**dict([[i,1]]),**func_fitted_kwargs)-constant) for i in linear_paramNames]
			
			###Now add them together to get result for each combo
			if len(nonlin_kys) == 0:	###I.e. only 1 combo was passed through each linear term calculation and constant
				result = constant
				for l in range(len(linear_paramNames)):
					result = result + linear_terms[l]*linears[l]
			else:
				constant = constant[...,nonlinears_unique_inverse]
				result = constant
				for l in range(len(linear_paramNames)):
					result = result + linear_terms[l][...,nonlinears_unique_inverse]*linears[l]
		else:
			result = func_fitted(x_cont_vals[...,None], **fit_param_combos, **func_fitted_kwargs)
	else:
		result = func_fitted(x_cont_vals[...,None], **fit_param_combos, **func_fitted_kwargs)
	# print(result.shape)
	mins = result.min(axis=1)
	maxes = result.max(axis=1)
	return mins, maxes

