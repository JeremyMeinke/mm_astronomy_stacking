#ACT_stack.py
###For stack on latest act (dr5) maps
from __future__ import division
import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import pandas as pd
import scipy.ndimage as spi
import time
import os
import healpy as hp
import astropy
from astropy.io import fits

def gnomonic(center_dec,center_ra,stepsize,sidesize=60):		###[Deg,Deg,Arcmin,Arcmin]
	theta=(90-center_dec)*np.pi/180			###[Rad]
	phi=center_ra*np.pi/180					###[Rad]

	rad=sidesize/2
	da=stepsize
	base=np.arange(-rad,rad+da,da)*np.pi/180/60		###convert to Rad here to make it stop doing dumb rounding Sh!t
	dr=len(base)
	base[int(np.rint((dr-1)/2))]=0
	x=base*np.ones([dr,dr])
	y=(base*np.ones([dr,dr])).T[::-1]		###because latitude number goes the other direction....took me wayyyy too long to realize this....
	rho=np.sqrt(x**2+y**2)
	c=np.arctan2(rho,1)
	lat=np.pi/2-theta
	t_set=np.pi/2-np.arcsin(np.cos(c)*np.sin(lat)+y*np.cos(c)*np.cos(lat))
	p_set=phi+np.arctan2(x*np.sin(c),rho*np.cos(lat)*np.cos(c)-y*np.sin(lat)*np.sin(c))
	return t_set,p_set

###Taken from CMB_School_Part_03
def make_2d_gaussian_beam(N,pix_size,fwhm):
     # make a 2d coordinate system
    N=int(N)
    ones = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.) * pix_size
    X = np.outer(ones,inds)
    Y = np.transpose(X)
    R = np.sqrt(X**2. + Y**2.)
    #plt.title('Radial coordinates')
    #plt.imshow(R)
  
    # make a 2d gaussian 
    beam_sigma = fwhm / np.sqrt(8.*np.log(2))
    gaussian = np.exp(-.5 *(R/beam_sigma)**2.)
    gaussian = gaussian / np.sum(gaussian)
    # return the gaussian
    #plt.imshow(gaussian)
    return(gaussian)

def GFilter(Image, pixres, High_fwhm, Low_fwhm):			##fwhm and pixres in same units
	f2s = 1 / (2 * np.sqrt(np.log(4)) )	
	high_sig = f2s*High_fwhm/pixres
	if High_fwhm != 0:
		remove = spi.gaussian_filter(Image, sigma=high_sig)
		new = Image-remove
	else:
		new = np.array(Image)
		
	low_sig = f2s*Low_fwhm/pixres
	if Low_fwhm != 0:
		new = spi.gaussian_filter(new, sigma=low_sig)
	
	return new

def change_gaussian_beam_map(Map,pix_size,map_beam_fwhm,new_beam_fwhm,both=True):
	"deconvolves a map with a Gaussian beam pattern.  NOTE: pix_size and beam_fwhm need to be in the same units" 
	
	###Plot Checking
	# print(new_FT_gaussian.shape)
	# plt.imshow(np.real(new_FT_gaussian/map_FT_gaussian))
	# plt.imshow(np.real(np.fft.fftshift(new_FT_gaussian)))
	# plt.show()
	
	###NEW NEW METHOD
	if map_beam_fwhm>new_beam_fwhm:
		# make a 2d gaussian of the map beam
		map_gaussian = make_2d_gaussian_beam(len(Map),pix_size,map_beam_fwhm)	###Since needs to match size of Map
		new_gaussian = make_2d_gaussian_beam(len(Map),pix_size,new_beam_fwhm)	###Since needs to match size of Map
		# do the convolution
		map_FT_gaussian=np.fft.fft2(np.fft.fftshift(map_gaussian)) # first add the shift so that it isredeconvolved-rstacks[f] central
		new_FT_gaussian=np.fft.fft2(np.fft.fftshift(new_gaussian)) # first add the shift so that it is central
		g_val=0.25
		if both:			###Can't decide if cutting both off at the same frequency location or at 1/sigma location is better
			new_FT_gaussian[map_FT_gaussian<g_val]=np.min(np.real(new_FT_gaussian[map_FT_gaussian>g_val]))
		else:
			new_FT_gaussian[map_FT_gaussian<g_val]=g_val
		
		map_FT_gaussian[map_FT_gaussian<g_val]=g_val
		FT_Map=np.fft.fft2(np.fft.fftshift(Map)) #shift the map too
		fft_final=FT_Map*np.real(new_FT_gaussian)/np.real(map_FT_gaussian)
		# print(np.max(fft_final))
		deconvolved_map=np.fft.fftshift(np.real(np.fft.ifft2(fft_final)))
	else:
		deconvolved_map=GFilter(Map,pix_size,0,np.sqrt(new_beam_fwhm**2-map_beam_fwhm**2))
	
	return deconvolved_map

###new method using my gnomonic project scheme (same as reproject.thumbnails(...,order=0,oversampling=1,...), but 2-5x faster)
def Stack(mapfile,catfile,side_arcmin,res,datasavename,alm_T_map_F=False,nside=8192,planck=False):	###[Arcmin,Arcmin], set planck=True if using planck maps (need to load differently and change coordinate system).  nside only needed to be provided if alm_T_map_F=True
	'''alm_T_map_F only used when planck=False, 
	nside only used if alm_T_map_F==True'''
	print('StartTime: ',time.asctime())
	###Getting catalog setup (takes decs and ras in radians)
	if catfile[-3:]=='txt':
		cat = np.genfromtxt(catfile) #Designed for array file with inner array for each galaxy/cluster, indices 1 and 2 are RA and DEC, respectively (where I have index 0 as galaxy #)
		if len(cat.shape)==1:
			clen=1
			decs=[cat[2]]
			ras=[cat[1]]
		else:
			clen=len(cat)
			decs=cat[:,2]
			ras=cat[:,1]
		cat=None;del cat	###To free up memory??
	elif catfile[-3:]=='pkl':
		cat=pd.read_pickle(catfile)
		clen=len(cat)
		ras=cat.iloc[:,0].to_numpy()
		decs=cat.iloc[:,1].to_numpy()
		cat=None;del cat	###To free up memory??
	else:
		print('Catalog file neither .txt or .pkl (pandas dataframe)...')
		return None
	###CHECK
	print(clen)
	print(np.max(decs),np.min(decs),np.max(ras),np.min(ras))	#Another validation check

	if planck:		###Since I often have to save them as separate fits files that healpy sometimes doesn't like
		m = astropy.io.fits.open(mapfile)
		mapdata = np.asarray(np.ndarray.tolist(m[1].data))
		print(mapdata.shape)
		mapdata=mapdata.reshape(np.prod(mapdata.shape),)	###Using np.prod so don't care about NSIDE anymore
		nside=int(np.sqrt(len(mapdata)/12))
		print('Planck Map, nside of %s predicted'%nside)
		m = None; del m
		r = hp.rotator.Rotator(coord=['G','C'])		###Since planck maps are often provided in galactic...whoopsie was rotating it backwards ['G','C'] before....
		mapdata=r.rotate_map_alms(mapdata)
		print(mapdata.shape)
	else:
		if alm_T_map_F:
			alm=hp.read_alm(mapfile)
			mapdata=hp.alm2map(alm, nside)
			alm=None; del alm
		else:
			mapdata=hp.read_map(mapfile)
			nside=int(np.sqrt(len(mapdata)/12))
			print('Healpy.read_map used, nside of %s predicted'%nside)

	stackset=np.zeros([int(np.round(side_arcmin/res)+1),int(np.round(side_arcmin/res)+1)])
	
	for c in range(clen):
		t_set,p_set=gnomonic(decs[c],ras[c],res,sidesize=side_arcmin)
		stackset+=mapdata[hp.ang2pix(nside,np.abs(t_set),p_set)]
	print(clen)
	savefile='temp/'+datasavename+'_%.2farcmin_%.2fres.txt'%(side_arcmin,res)
	np.savetxt(savefile,stackset, header='timestamp=%s'%str(time.asctime()))
	print('savename: ',savefile)
	print('PostStack check ',time.asctime())
	return None

def Stack_interpolate(mapfile,catfile,side_arcmin,res,datasavename,alm_T_map_F=False,nside=8192,planck=False):	###[Arcmin,Arcmin], set planck=True if using planck maps (need to load differently and change coordinate system).  nside only needed to be provided if alm_T_map_F=True
	'''alm_T_map_F only used when planck=False, 
	nside only used if alm_T_map_F==True'''
	print('StartTime: ',time.asctime())
	###Getting catalog setup (takes decs and ras in radians)
	if catfile[-3:]=='txt':
		cat = np.genfromtxt(catfile) #Designed for array file with inner array for each galaxy/cluster, indices 1 and 2 are RA and DEC, respectively (where I have index 0 as galaxy #)
		if len(cat.shape)==1:
			clen=1
			decs=[cat[2]]
			ras=[cat[1]]
		else:
			clen=len(cat)
			decs=cat[:,2]
			ras=cat[:,1]
		cat=None;del cat	###To free up memory??
	elif catfile[-3:]=='pkl':
		cat=pd.read_pickle(catfile)
		clen=len(cat)
		ras=cat.iloc[:,0].to_numpy()
		decs=cat.iloc[:,1].to_numpy()
		cat=None;del cat	###To free up memory??
	else:
		print('Catalog file neither .txt or .pkl (pandas dataframe)...')
		return None
	###CHECK
	print(clen)
	print(np.max(decs),np.min(decs),np.max(ras),np.min(ras))	#Another validation check

	if planck:		###Since I often have to save them as separate fits files that healpy sometimes doesn't like
		m = astropy.io.fits.open(mapfile)
		mapdata = np.asarray(np.ndarray.tolist(m[1].data))
		print(mapdata.shape)
		mapdata=mapdata.reshape(np.prod(mapdata.shape),)	###Using np.prod so don't care about NSIDE anymore
		nside=int(np.sqrt(len(mapdata)/12))
		print('Planck Map, nside of %s predicted'%nside)
		m = None; del m
		r = hp.rotator.Rotator(coord=['G','C'])		###Since planck maps are often provided in galactic...whoopsie was rotating it backwards ['G','C'] before....
		mapdata=r.rotate_map_alms(mapdata)
		print(mapdata.shape)
	else:
		if alm_T_map_F:
			alm=hp.read_alm(mapfile)
			mapdata=hp.alm2map(alm, nside)
			alm=None; del alm
		else:
			mapdata=hp.read_map(mapfile)
			nside=int(np.sqrt(len(mapdata)/12))
			print('Healpy.read_map used, nside of %s predicted'%nside)

	stackset=np.zeros([int(np.round(side_arcmin/res)+1),int(np.round(side_arcmin/res)+1)])
	
	for c in range(clen):
		t_set,p_set=gnomonic(decs[c],ras[c],res,sidesize=side_arcmin)
		stackset+=hp.get_interp_val(mapdata,np.abs(t_set),p_set)
	print(clen)
	savefile='temp/'+datasavename+'_interpolated_%.2farcmin_%.2fres.txt'%(side_arcmin,res)
	np.savetxt(savefile,stackset, header='timestamp=%s'%str(time.asctime()))
	print('savename: ',savefile)
	print('PostStack check ',time.asctime())
	return None

def CAP_Stack(mapfile,catfile,CAP_R_set,res,datasavename,alm_T_map_F=False,nside=8192,planck=False,version=0):	###[Arcmin,Arcmin]
	print('StartTime: ',time.asctime())
	###Getting catalog setup (takes decs and ras in radians)
	if catfile[-3:]=='txt':
		cat = np.genfromtxt(catfile) #Designed for array file with inner array for each galaxy/cluster, indices 1 and 2 are RA and DEC, respectively (where I have index 0 as galaxy #)
		if len(cat.shape)==1:
			clen=1
			decs=[cat[2]]
			ras=[cat[1]]
		else:
			clen=len(cat)
			decs=cat[:,2]
			ras=cat[:,1]
		cat=None;del cat	###To free up memory??
	elif catfile[-3:]=='pkl':
		cat=pd.read_pickle(catfile)
		clen=len(cat)
		ras=cat.iloc[:,0].to_numpy()
		decs=cat.iloc[:,1].to_numpy()
		cat=None;del cat	###To free up memory??
	else:
		print('Catalog file neither .txt or .pkl (pandas dataframe)...')
		return None

	###CHECK
	print(clen)
	print(np.max(decs),np.min(decs),np.max(ras),np.min(ras))	#Another validation check

	if planck:		###Since I often have to save them as separate fits files that healpy sometimes doesn't like
		m = astropy.io.fits.open(mapfile)
		mapdata = np.asarray(np.ndarray.tolist(m[1].data))
		mapdata=mapdata.reshape(np.prod(mapdata.shape),)	###Using np.prod so don't care about NSIDE anymore
		nside=int(np.sqrt(len(mapdata)/12))
		print('Planck Map, nside of %s predicted'%nside)
		m = None; del m
		r = hp.rotator.Rotator(coord=['G','C'])		###Since planck maps are often provided in galactic
		mapdata=r.rotate_map_alms(mapdata)
		print(mapdata.shape)
	else:
		if alm_T_map_F:
			alm=hp.read_alm(mapfile)
			mapdata=hp.alm2map(alm, nside)
		else:
			mapdata=hp.read_map(mapfile)
			nside=int(np.sqrt(len(mapdata)/12))
			print('Healpy.read_map used, nside of %s predicted'%nside)
	

	side_arcmin=3*np.max(CAP_R_set)
	print('side arc check: ',side_arcmin)
	DR=int(np.rint(side_arcmin/res+1))
	
	def CAP_Aps(CAP_R):
		CAP_kern = np.zeros([DR,DR])
		X,Y=np.meshgrid(np.arange(DR),np.arange(DR))
		dist=np.sqrt((X-int(np.rint((DR-1)/2)))**2 + (Y-int(np.rint((DR-1)/2)))**2)*res
		CAP_kern[dist<=np.sqrt(2)*CAP_R]=-res**2
		CAP_kern[dist<=CAP_R]=res**2
		return CAP_kern

	t_s=time.time()
	AP_set=[]
	for r in CAP_R_set:
		AP_set.append(CAP_Aps(r))
	print(AP_set[0].shape,len(AP_set))
	print('CAP create time: %s seconds for %i CAPs'%(time.time()-t_s,len(CAP_R_set))) ###TIME CHECK
	# t_s=time.time()
	# load_time=[]
	AP_vals=[]
	for c in range(clen)[:20]:
		# lt=time.time()
		t_set,p_set=gnomonic(decs[c],ras[c],res,sidesize=side_arcmin)
		stackset=mapdata[hp.ang2pix(nside,np.abs(t_set),p_set)]		###np.pi/2 bc mapdata.at requires declination input (not theta)
		# load_time.append(time.time()-lt)
		temp=[]
		for a in AP_set:
			temp.append(np.sum(stackset*a))	###creating list 'row'
		AP_vals.append(temp)	###adding completed 'row' to master list
	AP_vals=pd.DataFrame(AP_vals,columns=['%.3f'%CAP_R_set[i]+ ' [Arcmin]' for i in range(len(CAP_R_set))])		###Switching to pandas array, old version == #np.asarray(AP_vals)
	
	# ft=time.time()
	# print('Total CUTOUT and CAP time: %s seconds per Galaxy'%((ft-t_s)/20))
	# print('Average map CUTOUT ONLY time: %s seconds per Galaxy'%np.mean(load_time))
	# print('Average CAP ONLY time: %s seconds per Galaxy per CAP (avg)'%((ft-t_s)/20/len(CAP_R_set)-np.mean(load_time)/len(CAP_R_set)))
	# print(AP_vals)	###CHECK
	
	print(clen)

	####ensuring nothing is overwritten since data might differ
	k=version
	savefile='temp/'+datasavename+'_CAPs_%.2fres_v%i.pkl'%(res,k)
	while os.path.isfile(savefile):
		k+=1
		savefile='temp/'+datasavename+'_CAPs_%.2fres_v%i.pkl'%(res,k)
	print('savename: ',savefile)
	AP_vals.to_pickle(savefile,protocol=4)		###Save...comment out if not needed/testing
	print('PostStack check ',time.asctime())
	###OLD VERSION
	# np.savetxt(savefile,AP_vals,header='%s CAP Radiis,timestamp=%s'%(str(CAP_R_set),str(time.asctime())))
	return None

def CAP_Stack_ReBeam(mapfile,catfile,CAP_R_set,res,datasavename,old_fwhm,new_fwhm,alm_T_map_F=False,nside=8192,planck=False,version=0):	###[Arcmin,Arcmin]
	print('StartTime: ',time.asctime())
	###Getting catalog setup (takes decs and ras in radians)
	if catfile[-3:]=='txt':
		cat = np.genfromtxt(catfile) #Designed for array file with inner array for each galaxy/cluster, indices 1 and 2 are RA and DEC, respectively (where I have index 0 as galaxy #)
		if len(cat.shape)==1:
			clen=1
			decs=[cat[2]]
			ras=[cat[1]]
		else:
			clen=len(cat)
			decs=cat[:,2]
			ras=cat[:,1]
		cat=None;del cat	###To free up memory??
	elif catfile[-3:]=='pkl':
		cat=pd.read_pickle(catfile)
		clen=len(cat)
		ras=cat.iloc[:,0].to_numpy()
		decs=cat.iloc[:,1].to_numpy()
		cat=None;del cat	###To free up memory??
	else:
		print('Catalog file neither .txt or .pkl (pandas dataframe)...')
		return None

	###CHECK
	print(clen)
	print(np.max(decs),np.min(decs),np.max(ras),np.min(ras))	#Another validation check

	if planck:		###Since I often have to save them as separate fits files that healpy sometimes doesn't like
		m = astropy.io.fits.open(mapfile)
		mapdata = np.asarray(np.ndarray.tolist(m[1].data))
		mapdata=mapdata.reshape(np.prod(mapdata.shape),)	###Using np.prod so don't care about NSIDE anymore
		nside=int(np.sqrt(len(mapdata)/12))
		print('Planck Map, nside of %s predicted'%nside)
		m = None; del m
		r = hp.rotator.Rotator(coord=['G','C'])		###Since planck maps are often provided in galactic
		mapdata=r.rotate_map_alms(mapdata)
		print(mapdata.shape)
	else:
		if alm_T_map_F:
			alm=hp.read_alm(mapfile)
			mapdata=hp.alm2map(alm, nside)
		else:
			mapdata=hp.read_map(mapfile)
			nside=int(np.sqrt(len(mapdata)/12))
			print('Healpy.read_map used, nside of %s predicted'%nside)


	side_arcmin=3*np.max(CAP_R_set)+3*np.max([old_fwhm,new_fwhm])		###Padding by 3* max fwhm to negate edge effects
	print('side arc check: ',side_arcmin)
	DR=int(np.rint(side_arcmin/res+1))
	stackset=np.zeros([DR,DR])
	
	def CAP_Aps(CAP_R):
		CAP_kern = np.full([DR,DR],np.nan)
		for x in range(DR):
			for y in range(DR):
				dist = np.sqrt((x-int(np.rint((DR-1)/2)))**2 + (y-int(np.rint((DR-1)/2)))**2 )
				if dist <= CAP_R/res:
					CAP_kern[x,y] = res**2
				elif dist<=np.sqrt(2)*CAP_R/res:
					CAP_kern[x,y] = -res**2
		return CAP_kern

	t_s=time.time()
	AP_set=[]
	for r in CAP_R_set:
		AP_set.append(CAP_Aps(r))
	print(AP_set[0].shape,len(AP_set))
	print('CAP create time: %s seconds for %i CAPs'%(time.time()-t_s,len(CAP_R_set))) ###TIME CHECK
	# t_s=time.time()
	# load_time=[]
	AP_vals=[]
	for c in range(clen):
		# lt=time.time()
		t_set,p_set=gnomonic(decs[c],ras[c],res,sidesize=side_arcmin)
		stackset=mapdata[hp.ang2pix(nside,np.abs(t_set),p_set)]
		stackset=change_gaussian_beam_map(stackset,res,old_fwhm,new_fwhm)
		# load_time.append(time.time()-lt)
		temp=[]
		for a in AP_set:
			temp.append(np.nansum(stackset*a))	###creating list 'row'
		AP_vals.append(temp)	###adding completed 'row' to master list
	AP_vals=pd.DataFrame(AP_vals,columns=['%.3f'%CAP_R_set[i]+ ' [Arcmin]' for i in range(len(CAP_R_set))])		###Switching to pandas array, old version == #np.asarray(AP_vals)
	
	# ft=time.time()
	# print('Total CUTOUT and CAP time: %s seconds per Galaxy'%((ft-t_s)/20))
	# print('Average map CUTOUT ONLY time: %s seconds per Galaxy'%np.mean(load_time))
	# print('Average CAP ONLY time: %s seconds per Galaxy per CAP (avg)'%((ft-t_s)/20/len(CAP_R_set)-np.mean(load_time)/len(CAP_R_set)))
	# print(AP_vals)	###CHECK
	
	print(clen)

	####ensuring nothing is overwritten since data might differ
	k=version
	savefile='temp/'+datasavename+'_CAPs_%.2ffwhm_%.2fres_v%i.pkl'%(new_fwhm,res,k)
	while os.path.isfile(savefile):
		k+=1
		savefile='temp/'+datasavename+'_CAPs_%.2ffwhm_%.2fres_v%i.pkl'%(new_fwhm,res,k)
	print('savename: ',savefile)
	AP_vals.to_pickle(savefile,protocol=4)		###Save...comment out if not needed/testing
	print('PostStack check ',time.asctime())
	###OLD VERSION
	# np.savetxt(savefile,AP_vals,header='%s CAP Radiis,timestamp=%s'%(str(CAP_R_set),str(time.asctime())))
	return None

def Stack_radial_avg(mapfile,catfile,rad_edges,res,datasavename,alm_T_map_F=False,nside=8192,planck=False,version=0):		###[Arcmin, Arcmin]
	'''Where rad_edges==list or numpy.ndarray of radial edges, i.e. radial averages taken between those values,
	edges expected at no less than .1 intervals (%.1f column headers saved as)'''
	print('StartTime: ',time.asctime())
	###Getting catalog setup (takes decs and ras in radians)
	if catfile[-3:]=='txt':
		cat = np.genfromtxt(catfile) #Designed for array file with inner array for each galaxy/cluster, indices 1 and 2 are RA and DEC, respectively (where I have index 0 as galaxy #)
		if len(cat.shape)==1:
			clen=1
			decs=[cat[2]]
			ras=[cat[1]]
		else:
			clen=len(cat)
			decs=cat[:,2]
			ras=cat[:,1]
		cat=None;del cat	###To free up memory??
	elif catfile[-3:]=='pkl':
		cat=pd.read_pickle(catfile)
		clen=len(cat)
		ras=cat.iloc[:,0].to_numpy()
		decs=cat.iloc[:,1].to_numpy()
		cat=None;del cat	###To free up memory??
	else:
		print('Catalog file neither .txt or .pkl (pandas dataframe)...')
		return None
	
	if planck:		###Since I often have to save them as separate fits files that healpy sometimes doesn't like
		m = astropy.io.fits.open(mapfile)
		mapdata = np.asarray(np.ndarray.tolist(m[1].data))
		print(mapdata.shape)
		mapdata=mapdata.reshape(np.prod(mapdata.shape),)	###Using np.prod so don't care about NSIDE anymore
		nside=int(np.sqrt(len(mapdata)/12))
		print('Planck Map, nside of %s predicted'%nside)
		m = None; del m
		r = hp.rotator.Rotator(coord=['G','C'])		###Since planck maps are often provided in galactic
		mapdata=r.rotate_map_alms(mapdata)
		print(mapdata.shape)
	else:
		if alm_T_map_F:
			alm=hp.read_alm(mapfile)
			mapdata=hp.alm2map(alm, nside)
		else:
			mapdata=hp.read_map(mapfile)
			nside=int(np.sqrt(len(mapdata)/12))
			print('Healpy.read_map used, nside of %s predicted'%nside)

	def rad_avgs(rmin,rmax):
		radavg_kern = np.full([DR,DR],np.nan)
		X,Y=np.meshgrid(np.arange(DR),np.arange(DR))
		dist=np.sqrt((X-int(np.rint((DR-1)/2)))**2 + (Y-int(np.rint((DR-1)/2)))**2)*res
		radavg_kern[(dist>=rmin)&(dist<rmax)]=1		###Changing values equal/above rmin (to include 0), and below rmax
		return radavg_kern

	side_arcmin=np.max(rad_edges)*2+2
	DR=int(np.round(side_arcmin/res)+1)
	stackset=np.zeros([DR,DR])
	stacklen=len(stackset)
	print(stacklen,stackset.shape)
	rad_avg_set=[rad_avgs(rad_edges[r],rad_edges[r+1]) for r in range(len(rad_edges)-1)]
	dataset=[]
	h=['%.2f to %.2f [Arcmin]'%(rad_edges[r],rad_edges[r+1]) for r in range(len(rad_edges)-1)]

	for c in range(clen):
		t_set,p_set=gnomonic(decs[c],ras[c],res,sidesize=side_arcmin)
		stackset=mapdata[hp.ang2pix(nside,np.abs(t_set),p_set)]
		stackset-=np.mean(stackset)				###Subtracting the mean
		dataset.append([])
		for rad in rad_avg_set:
			dataset[c].append(np.nanmean(rad*stackset))
	print(clen)
	DF=pd.DataFrame(dataset,columns=h)	
	# print(DF)
	
	####ensuring nothing is overwritten since data might differ
	k=version
	savefile='temp/'+datasavename+'_radavg_%.2fres_v%i.pkl'%(res,k)
	while os.path.isfile(savefile):
		k+=1
		savefile='temp/'+datasavename+'_radavg_%.2fres_v%i.pkl'%(res,k)
	print('savename: ',savefile)
	DF.to_pickle(savefile,protocol=4)		###Save...comment out if not needed/testing
	print('PostStack check ',time.asctime())
	return None

def Stack_radial_avg_interpolate(mapfile,catfile,rad_edges,res,datasavename,alm_T_map_F=False,nside=8192,planck=False,version=0):		###[Arcmin, Arcmin]
	'''Where rad_edges==list or numpy.ndarray of radial edges, i.e. radial averages taken between those values,
	edges expected at no less than .1 intervals (%.1f column headers saved as)'''
	print('StartTime: ',time.asctime())
	###Getting catalog setup (takes decs and ras in radians)
	if catfile[-3:]=='txt':
		cat = np.genfromtxt(catfile) #Designed for array file with inner array for each galaxy/cluster, indices 1 and 2 are RA and DEC, respectively (where I have index 0 as galaxy #)
		if len(cat.shape)==1:
			clen=1
			decs=[cat[2]]
			ras=[cat[1]]
		else:
			clen=len(cat)
			decs=cat[:,2]
			ras=cat[:,1]
		cat=None;del cat	###To free up memory??
	elif catfile[-3:]=='pkl':
		cat=pd.read_pickle(catfile)
		clen=len(cat)
		ras=cat.iloc[:,0].to_numpy()
		decs=cat.iloc[:,1].to_numpy()
		cat=None;del cat	###To free up memory??
	else:
		print('Catalog file neither .txt or .pkl (pandas dataframe)...')
		return None
	
	if planck:		###Since I often have to save them as separate fits files that healpy sometimes doesn't like
		m = astropy.io.fits.open(mapfile)
		mapdata = np.asarray(np.ndarray.tolist(m[1].data))
		print(mapdata.shape)
		mapdata=mapdata.reshape(np.prod(mapdata.shape),)	###Using np.prod so don't care about NSIDE anymore
		nside=int(np.sqrt(len(mapdata)/12))
		print('Planck Map, nside of %s predicted'%nside)
		m = None; del m
		r = hp.rotator.Rotator(coord=['G','C'])		###Since planck maps are often provided in galactic
		mapdata=r.rotate_map_alms(mapdata)
		print(mapdata.shape)
	else:
		if alm_T_map_F:
			alm=hp.read_alm(mapfile)
			mapdata=hp.alm2map(alm, nside)
		else:
			mapdata=hp.read_map(mapfile)
			nside=int(np.sqrt(len(mapdata)/12))
			print('Healpy.read_map used, nside of %s predicted'%nside)

	def rad_avgs(rmin,rmax):
		radavg_kern = np.full([DR,DR],np.nan)
		X,Y=np.meshgrid(np.arange(DR),np.arange(DR))
		dist=np.sqrt((X-int(np.rint((DR-1)/2)))**2 + (Y-int(np.rint((DR-1)/2)))**2)*res
		radavg_kern[(dist>=rmin)&(dist<rmax)]=1		###Changing values equal/above rmin (to include 0), and below rmax
		return radavg_kern

	side_arcmin=np.max(rad_edges)*2+2
	DR=int(np.round(side_arcmin/res)+1)
	stackset=np.zeros([DR,DR])
	stacklen=len(stackset)
	print(stacklen,stackset.shape)
	rad_avg_set=[rad_avgs(rad_edges[r],rad_edges[r+1]) for r in range(len(rad_edges)-1)]
	dataset=[]
	h=['%.2f to %.2f [Arcmin]'%(rad_edges[r],rad_edges[r+1]) for r in range(len(rad_edges)-1)]

	for c in range(clen):
		t_set,p_set=gnomonic(decs[c],ras[c],res,sidesize=side_arcmin)
		stackset=hp.get_interp_val(mapdata,np.abs(t_set),p_set)
		stackset-=np.mean(stackset)				###Subtracting the mean
		dataset.append([])
		for rad in rad_avg_set:
			dataset[c].append(np.nanmean(rad*stackset))
	print(clen)
	DF=pd.DataFrame(dataset,columns=h)	
	# print(DF)
	
	####ensuring nothing is overwritten since data might differ
	k=version
	savefile='temp/'+datasavename+'_interpolated_radavg_%.2fres_v%i.pkl'%(res,k)
	while os.path.isfile(savefile):
		k+=1
		savefile='temp/'+datasavename+'_interpolated_radavg_%.2fres_v%i.pkl'%(res,k)
	print('savename: ',savefile)
	DF.to_pickle(savefile,protocol=4)		###Save...comment out if not needed/testing
	print('PostStack check ',time.asctime())
	return None

###Radial Average of annulus between rad_edges, after beam correction and mean subtraction
def Stack_radial_avg_ReBeam(mapfile,catfile,rad_edges,res,datasavename,old_fwhm,new_fwhm,alm_T_map_F=False,nside=8192,planck=False,version=0):		###[Arcmin, Arcmin]
	'''Where rad_edges==list or numpy.ndarray of radial edges, i.e. radial averages taken between those values, 
	edges expected at no less than .1 intervals (%.1f column headers saved as)'''
	print('StartTime: ',time.asctime())
	###Getting catalog setup (takes decs and ras in radians)
	if catfile[-3:]=='txt':
		cat = np.genfromtxt(catfile) #Designed for array file with inner array for each galaxy/cluster, indices 1 and 2 are RA and DEC, respectively (where I have index 0 as galaxy #)
		if len(cat.shape)==1:
			clen=1
			decs=[cat[2]]
			ras=[cat[1]]
		else:
			clen=len(cat)
			decs=cat[:,2]
			ras=cat[:,1]
		cat=None;del cat	###To free up memory??
	elif catfile[-3:]=='pkl':
		cat=pd.read_pickle(catfile)
		clen=len(cat)
		ras=cat.iloc[:,0].to_numpy()
		decs=cat.iloc[:,1].to_numpy()
		cat=None;del cat	###To free up memory??
	else:
		print('Catalog file neither .txt or .pkl (pandas dataframe)...')
		return None
	
	if planck:		###Since I often have to save them as separate fits files that healpy sometimes doesn't like
		m = astropy.io.fits.open(mapfile)
		mapdata = np.asarray(np.ndarray.tolist(m[1].data))
		print(mapdata.shape)
		mapdata=mapdata.reshape(np.prod(mapdata.shape),)	###Using np.prod so don't care about NSIDE anymore
		nside=int(np.sqrt(len(mapdata)/12))
		print('Planck Map, nside of %s predicted'%nside)
		m = None; del m
		r = hp.rotator.Rotator(coord=['G','C'])		###Since planck maps are often provided in galactic
		mapdata=r.rotate_map_alms(mapdata)
		print(mapdata.shape)
	else:
		if alm_T_map_F:
			alm=hp.read_alm(mapfile)
			mapdata=hp.alm2map(alm, nside)
		else:
			mapdata=hp.read_map(mapfile)
			nside=int(np.sqrt(len(mapdata)/12))
			print('Healpy.read_map used, nside of %s predicted'%nside)

	def rad_avgs(rmin,rmax):
		radavg_kern = np.full([DR,DR],np.nan)
		X,Y=np.meshgrid(np.arange(DR),np.arange(DR))
		dist=np.sqrt((X-int(np.rint((DR-1)/2)))**2 + (Y-int(np.rint((DR-1)/2)))**2)*res
		radavg_kern[(dist>=rmin)&(dist<rmax)]=1		###Changing values equal/above rmin (to include 0), and below rmax
		return radavg_kern

	side_arcmin=np.max(rad_edges)*2+2+3*np.max([old_fwhm,new_fwhm])
	DR=int(np.round(side_arcmin/res)+1)
	stackset=np.zeros([DR,DR])
	stacklen=len(stackset)
	print(stacklen,stackset.shape)
	rad_avg_set=[rad_avgs(rad_edges[r],rad_edges[r+1]) for r in range(len(rad_edges)-1)]
	dataset=[]
	h=['%.2f to %.2f [Arcmin]'%(rad_edges[r],rad_edges[r+1]) for r in range(len(rad_edges)-1)]

	for c in range(clen):
		t_set,p_set=gnomonic(decs[c],ras[c],res,sidesize=side_arcmin)
		stackset=mapdata[hp.ang2pix(nside,np.abs(t_set),p_set)]
		stackset=change_gaussian_beam_map(stackset,res,old_fwhm,new_fwhm)
		stackset-=np.mean(stackset)				###Subtracting the mean after beam change
		dataset.append([])
		for rad in rad_avg_set:
			dataset[c].append(np.nanmean(rad*stackset))
	print(clen)
	DF=pd.DataFrame(dataset,columns=h)	
	# print(DF)
	
	####ensuring nothing is overwritten since data might differ
	k=version
	savefile='temp/'+datasavename+'_radavg_%.2ffwhm_%.2fres_v%i.pkl'%(new_fwhm,res,k)
	while os.path.isfile(savefile):
		k+=1
		savefile='temp/'+datasavename+'_radavg_%.2ffwhm_%.2fres_v%i.pkl'%(new_fwhm,res,k)
	print('savename: ',savefile)
	DF.to_pickle(savefile,protocol=4)		###Save...comment out if not needed/testing
	print('PostStack check ',time.asctime())
	return None


###Gives RMS back
def Stack_rms_only(mapfile,catfile,side_arcmins,res,datasavename,alm_T_map_F=False,nside=8192,planck=False,version=0):		###[Arcmin, Arcmin]
	'''nside only needed i alm_T'''
	print('StartTime: ',time.asctime())
	###Getting catalog setup (takes decs and ras in radians)
	if catfile[-3:]=='txt':
		cat = np.genfromtxt(catfile) #Designed for array file with inner array for each galaxy/cluster, indices 1 and 2 are RA and DEC, respectively (where I have index 0 as galaxy #)
		if len(cat.shape)==1:
			clen=1
			decs=[cat[2]]
			ras=[cat[1]]
		else:
			clen=len(cat)
			decs=cat[:,2]
			ras=cat[:,1]
		cat=None;del cat	###To free up memory??
	elif catfile[-3:]=='pkl':
		cat=pd.read_pickle(catfile)
		clen=len(cat)
		ras=cat.iloc[:,0].to_numpy()
		decs=cat.iloc[:,1].to_numpy()
		cat=None;del cat	###To free up memory??
	else:
		print('Catalog file neither .txt or .pkl (pandas dataframe)...')
		return None
	
	if planck:		###Since I often have to save them as separate fits files that healpy sometimes doesn't like
		m = astropy.io.fits.open(mapfile)
		mapdata = np.asarray(np.ndarray.tolist(m[1].data))
		print(mapdata.shape)
		mapdata=mapdata.reshape(np.prod(mapdata.shape),)	###Using np.prod so don't care about NSIDE anymore
		nside=int(np.sqrt(len(mapdata)/12))
		print('Planck Map, nside of %s predicted'%nside)
		m = None; del m
		r = hp.rotator.Rotator(coord=['G','C'])		###Since planck maps are often provided in galactic
		mapdata=r.rotate_map_alms(mapdata)
		print(mapdata.shape)
	else:
		if alm_T_map_F:
			alm=hp.read_alm(mapfile)
			mapdata=hp.alm2map(alm, nside)
		else:
			mapdata=hp.read_map(mapfile)
			nside=int(np.sqrt(len(mapdata)/12))
			print('Healpy.read_map used, nside of %s predicted'%nside)

	if isinstance(side_arcmins,list):		###If given a list of side_arcmins, set side_arcmin as the max (since we can grab largest stamp size and then calculate rms of all side_arcmins from within that stamp)
		print('list of side_arcs: ',side_arcmins)
		side_arcmin=np.max(side_arcmins)
		arclist=True 	###i.e. side_arcmins given as a list
		h=['%.3f [Arcmin]'%a for a in np.sort(side_arcmins)]
	else:
		side_arcmin=side_arcmins
		arclist=False
		h=['%.3f [Arcmin]'%side_arcmin]
	
	DR=int(np.round(side_arcmin/res)+1)
	stackset=np.zeros([DR,DR])
	stacklen=len(stackset)
	print(stacklen,stackset.shape)
	l=np.arange(DR)
	R=np.sqrt(np.sum(np.stack(np.array([*np.meshgrid(l-np.mean(l),l-np.mean(l))])**2),axis=0))
	###Quick check
	# plt.imshow(R)
	# plt.savefig(time.strftime("RMS_test_on%Y-%m-%d_at%H_%M.png"))
	# plt.close()

	dataset=[]
	if arclist:
		for c in range(clen):
			t_set,p_set=gnomonic(decs[c],ras[c],res,sidesize=side_arcmin)
			stackset=mapdata[hp.ang2pix(nside,np.abs(t_set),p_set)]
			dataset.append([])
			for a in np.sort(side_arcmins):
				dataset[c].append(np.std(stackset[R<=a]))
				###Old way
				# index=int((side_arcmin-a)/2/res)		###(+1 cancels out on difference)
				# dataset[c].append(np.std(stackset[index:stacklen-index]))
	else:
		for c in range(clen):
			t_set,p_set=gnomonic(decs[c],ras[c],res,sidesize=side_arcmin)
			stackset=mapdata[hp.ang2pix(nside,np.abs(t_set),p_set)]
			dataset.append(np.std(stackset[R<=side_arcmin]))
	print(clen)
	DF=pd.DataFrame(dataset,columns=h)	
	# print(DF)
	
	####ensuring nothing is overwritten since data might differ
	k=version
	savefile='temp/'+datasavename+'_RMS_%.2fres_v%i.pkl'%(res,k)
	while os.path.isfile(savefile):
		k+=1
		savefile='temp/'+datasavename+'_RMS_%.2fres_v%i.pkl'%(res,k)
	print('savename: ',savefile)
	DF.to_pickle(savefile,protocol=4)		###Save...comment out if not needed/testing, protocol=4 for back-compatability
	print('PostStack check ',time.asctime())
	return None

####Old method (with reproject.thumbnails)
# def Stack(mapfile,catfile,side_arcmin,res,datasavename):	###[Arcmin,Arcmin]
# 	print('StartTime: ',time.asctime())
# 	###Getting catalog setup (takes decs and ras in radians)
# 	cat = np.genfromtxt(catfile) #Designed for array file with inner array for each galaxy/cluster, indices 1 and 2 are RA and DEC, respectively (where I have index 0 as galaxy #)
# 	if len(cat.shape)==1:
# 		clen=1
# 		decs=[np.deg2rad(cat[2])]
# 		ras=[np.deg2rad(cat[1])]
# 	else:
# 		clen=len(cat)
# 		decs=np.deg2rad(cat[:,2])
# 		ras=np.deg2rad(cat[:,1])
# 	cat=None;del cat	###To free up memory??

# 	mapdata=enmap.read_map(mapfile)
# 	stackset=np.zeros([int(np.round(side_arcmin/res)+1),int(np.round(side_arcmin/res)+1)])
# 	side_r=np.deg2rad(side_arcmin/2/60)
# 	res=np.deg2rad(res/60)
	
# 	if stackset.shape != reproject.thumbnails(mapdata,coords=[decs[0],ras[0]],r=side_r,res=res).shape:
# 		print('STUPID PIXELL ARRAY SIZE AINT SQUARE')
# 		return None
	
# 	for c in range(clen):
# 		stackset+=reproject.thumbnails(mapdata,coords=[decs[c],ras[c]],r=side_r,res=res)
# 	print(clen)
# 	savefile='temp/'+datasavename+'_%.2farcmin_%.2fres.txt'%(side_arcmin,60*np.deg2rad(res))
# 	np.savetxt(savefile,stackset)
# 	print('PostStack check ',time.asctime())
# 	return None


###CAP SIZE TESTING
# res=0.05
# CAP_R_set=np.arange(1,6.21,0.2)
# side_arcmin=3*np.max(CAP_R_set)
# print('side arc check: ',side_arcmin)
# DR=int(np.rint(np.round(side_arcmin/res)+1))
# stackset=np.zeros([DR,DR])

# def CAP_Aps(CAP_R):
# 	CAP_kern = np.full([DR,DR],np.nan)
# 	for x in range(DR):
# 		for y in range(DR):
# 			dist = np.sqrt((x-int(np.rint((DR-1)/2)))**2 + (y-int(np.rint((DR-1)/2)))**2 )
# 			if dist <= CAP_R/res:
# 				CAP_kern[x,y] = res**2
# 			elif dist<=np.sqrt(2)*CAP_R/res:
# 				CAP_kern[x,y] = -res**2
# 	return CAP_kern

# print(np.max(CAP_R_set))
# plt.imshow(CAP_Aps(np.max(CAP_R_set)))
# plt.show()