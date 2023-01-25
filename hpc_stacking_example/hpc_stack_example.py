from __future__ import division
import numpy as np
import healpy as hp
import sys
import os.path
import ACT_stack	###Stack(almfile, NSIDE, search_arcmin, catalogfile, datasavename), CAP_Stack(mapfile,catfile,CAP_R_set,res,datasavename)
import Healpix_stack

###Master Variables for whatever used further below
n=int(sys.argv[1])
Freq = [90,150,220]
CAP_R_set=np.arange(1,6.21,0.2)
# rstep=0.1
# rad_avgs=np.arange(0,8.,rstep)		###Check This....
# rstep2=0.25
# # rad_avgs2=np.arange(0,10.1,rstep2)		###Check This....
# rad_avgs3=np.arange(0,20.1,rstep2)		###Check This....

new_rstep=0.50
new_rad_avg=np.arange(0,20.1,new_rstep)		###Check This...

arc=60		#[arcmin] stamp side size, aka arc x arc dimension image
res=0.05	##Default size for act pixell thumbnails (and coincidentally my chosen resolution for SPT stamps... so yay)
beam_fwhm=[2.1,1.3,1.0]		###arcmin, for ACT (SPT-SZ + Planck is 1.85 arcmin)
arc_set=[10,20,30,40,50,60]			###arcmin, for RMS calculations

###CAP Galaxies
# new_fwhm=2.1
# cname='./Whole_Catalogs/des_wise_act_select_clean_wclusters_v00_4arc.pkl'		###4arc nearby removal
# savename='actplanck5_f%s_T_des_highz_4arc_wclusters'%str(Freq[n]).zfill(3)		
# cname='./Whole_Catalogs/des_wise_act_select_wclusters_v00.pkl'		###No 4arc
# savename='actplanck5_f%s_T_des_highz_wclusters'%str(Freq[n]).zfill(3)		
# mapfile='act_planck_dr5.01_s08s18_AA_f%s_dn_map_act_T.fits'%str(Freq[n]).zfill(3)
# if os.path.isfile(cname) and os.path.isfile(mapfile):
# 	ACT_stack.CAP_Stack_ReBeam(mapfile=mapfile,catfile=cname,CAP_R_set=CAP_R_set,res=res,datasavename=savename,old_fwhm=beam_fwhm[n],new_fwhm=new_fwhm)
# else:
# 	print('No catalog or map file by the name: ', cname, mapfile)


###CAP Random (ACT)
# new_fwhm=2.1
# div=12
# f=Freq[n%3]
# d=n//3+1
# print(d,f)
# cname='./Whole_Catalogs/des_wise_rand_act_4arc_1075791_v0_div%iof12.pkl'%(d)			###4arc
# savename='actplanck5_f%s_T_des_rand_act_4arc_1075791_v0_div%iof12'%(str(f).zfill(3),d)
# cname='./Whole_Catalogs/des_wise_rand_act_1mil_v0_div%iof12.pkl'%(d)			###No 4arc
# savename='actplanck5_f%s_T_des_rand_act_1mil_v0_div%iof12'%(str(f).zfill(3),d)
# mapfile='act_planck_dr5.01_s08s18_AA_f%s_dn_map_act_T.fits'%str(f).zfill(3)
# if os.path.isfile(cname) and os.path.isfile(mapfile):
# 	ACT_stack.CAP_Stack_ReBeam(mapfile=mapfile,catfile=cname,CAP_R_set=CAP_R_set,res=res,datasavename=savename,old_fwhm=beam_fwhm[n%3],new_fwhm=new_fwhm)
# else:
# 	print('No catalog or map file by the name: ', cname, mapfile)


###rad avgs of Galaxies (smoothed all to new_fwhm)	###CHECK VERSION
# # new_fwhm=2.1
# # cname='./Whole_Catalogs/des_wise_act_select_clean_wclusters_v00_4arc.pkl'		###4arc nearby removal
# # savename='actplanck5_f%s_T_des_highz_4arc_wclusters'%str(Freq[n]).zfill(3)		
# cname='./Whole_Catalogs/des_wise_act_select_wclusters_v00.pkl'		###No 4arc

# ###With Sources ACT maps
# # savename='actplanck5_f%s_T_des_highz_wclusters'%str(Freq[n]).zfill(3)
# # mapfile='act_planck_dr5.01_s08s18_AA_f%s_dn_map_act_T.fits'%str(Freq[n]).zfill(3)

# ###Source-Free ACT maps
# # mapfile='act_planck_dr5.01_s08s18_AA_f%s_daynight_map_srcfree_T.fits'%str(Freq[n]).zfill(3)
# # mapfile="act_planck_dr5.01_s08s18_AA_f%s_daynight_map_srcfree_T_2.10fwhm_minus_SMICAnoSZ.fits"%str(Freq[n%3]).zfill(3)	###New Smoothed minus SMICAnoSZ
# # savename='actplanck5_f%s_srcfree_T_2.10fwhm_minus_SMICAnoSZ_des_highz_wclusters'%str(Freq[n]).zfill(3)
# # if os.path.isfile(cname) and os.path.isfile(mapfile):
# # 	ACT_stack.Stack_radial_avg(mapfile,cname,rad_avgs3,res,datasavename=savename)
# # 	# ACT_stack.Stack_radial_avg_ReBeam(mapfile,cname,rad_avgs3,res,datasavename=savename,old_fwhm=beam_fwhm[n],new_fwhm=new_fwhm,version=2)
# # else:
# # 	print('No catalog or map file by the name: ', cname, mapfile)

# ###Healpix alm (smoothed, noCMB, etc)
# mapfile="act_planck_dr5.01_s08s18_AA_f%s_daynight_map_srcfree_T_2.10fwhm_minus_SMICAnoSZ_Healpix8192_alm.fits"%str(Freq[n%3]).zfill(3)	###New Smoothed ACT minus SMICAnoSZ in Healpix8192 alm
# savename='actplanck5_f%s_srcfree_T_2.10fwhm_minus_SMICAnoSZ_Healpix8192_des_highz_wclusters'%str(Freq[n]).zfill(3)
# if os.path.isfile(cname) and os.path.isfile(mapfile):
# 	# Healpix_stack.Stack(mapfile=mapfile,catfile=cname,side_arcmin=arc,res=res,datasavename=savename,alm_T_map_F=True)
# 	Healpix_stack.Stack_radial_avg_interpolate(mapfile=mapfile,catfile=cname,rad_edges=new_rad_avg,res=res,datasavename=savename,alm_T_map_F=True,)	###Interpolated
# else:
# 	print('No catalog or map file by the name: ', cname, mapfile)



###rad avgs of Random (smoothed all to new_fwhm)	###CHECK VERSION
# new_fwhm=2.1
# div=12
# version=0
# f=Freq[n%3]
# d=n//3+1
# print(d,f)
# # cname='./Whole_Catalogs/des_wise_rand_act_4arc_1075791_v0_div%iof12.pkl'%(d)			###4arc
# # savename='actplanck5_f%s_T_des_rand_act_4arc_1075791_v0_div%iof12'%(str(f).zfill(3),d)
# cname='./Whole_Catalogs/des_wise_rand_act_1mil_v%i_div%iof12.pkl'%(version,d)			###No 4arc source filter

# ###With Sources
# # savename='actplanck5_f%s_T_des_rand_act_1mil_v%i_div%iof12'%(str(f).zfill(3),version,d)
# # mapfile='act_planck_dr5.01_s08s18_AA_f%s_dn_map_act_T.fits'%str(f).zfill(3)

# ###Source-Free version
# savename='actplanck5_f%s_srcfree_T_des_rand_act_1mil_v%i_div%iof12'%(str(f).zfill(3),version,d)
# mapfile='act_planck_dr5.01_s08s18_AA_f%s_daynight_map_srcfree_T.fits'%str(f).zfill(3)


# if os.path.isfile(cname) and os.path.isfile(mapfile):
# 	ACT_stack.Stack_radial_avg_ReBeam(mapfile,cname,rad_avgs2,res,datasavename=savename,old_fwhm=beam_fwhm[n%3],new_fwhm=new_fwhm,version=1)
# else:
# 	print('No catalog or map file by the name: ', cname, mapfile)



###rad avgs for Random SPT randoms (for comparison), using srcfree version
# # new_fwhm=2.1
# div=12
# f=Freq[n%3]
# d=n//3+1
# print(d,f)
# ###With Sources
# # mapfile='act_planck_dr5.01_s08s18_AA_f%s_dn_map_act_T.fits'%str(f).zfill(3)
# # savename='actplanck5_f%s_T_des_rand_spt_1mil_v%i_div%iof12'%(str(f).zfill(3),version,d)

# ###Source-Free version
# # mapfile='act_planck_dr5.01_s08s18_AA_f%s_daynight_map_srcfree_T.fits'%str(f).zfill(3)
# # savename='actplanck5_f%s_srcfree_T_des_rand_spt_1mil_v%i_div%iof12'%(str(f).zfill(3),version,d)

# ###Rand SPT############################3
# # version=1
# # cname='./Whole_Catalogs/des_wise_rand_spt_1mil_v%i_div%iof12.pkl'%(version,d)			###No 4arc source filter, spt
# # ###Healpix alm
# # mapfile="act_planck_dr5.01_s08s18_AA_f%s_daynight_map_srcfree_T_2.10fwhm_minus_SMICAnoSZ_Healpix8192_alm.fits"%str(f).zfill(3)	###New Smoothed ACT minus SMICAnoSZ in Healpix8192 alm
# # savename='actplanck5_f%s_srcfree_T_2.10fwhm_minus_SMICAnoSZ_Healpix8192_des_rand_spt_1mil_v%i_div%iof12'%(str(f).zfill(3),version,d)

# ###Rand ACT############################3
# version=0
# cname='./Whole_Catalogs/des_wise_rand_act_1mil_v%i_div%iof12.pkl'%(version,d)			###No 4arc source filter, spt
# ###Healpix alm
# mapfile="act_planck_dr5.01_s08s18_AA_f%s_daynight_map_srcfree_T_2.10fwhm_minus_SMICAnoSZ_Healpix8192_alm.fits"%str(f).zfill(3)	###New Smoothed ACT minus SMICAnoSZ in Healpix8192 alm
# savename='actplanck5_f%s_srcfree_T_2.10fwhm_minus_SMICAnoSZ_Healpix8192_des_rand_act_1mil_v%i_div%iof12'%(str(f).zfill(3),version,d)

# if os.path.isfile(cname) and os.path.isfile(mapfile):
# 	# ACT_stack.Stack_radial_avg_ReBeam(mapfile,cname,rad_avgs3,res,datasavename=savename,old_fwhm=beam_fwhm[n%3],new_fwhm=new_fwhm,version=2)
# 	Healpix_stack.Stack_radial_avg_interpolate(mapfile,cname,new_rad_avg,res,savename,alm_T_map_F=True)
# else:
# 	print('No catalog or map file by the name: ', cname, mapfile)

###eBOSS New SGC ELG catalog#########
# f=Freq[n%3]
# print(f)

# ###Healpix alm
# cname="./Whole_Catalogs/eBOSS_ELG_SGC-vDR16.pkl"		###465199 galaxies in ACT Field, no overlap with SPT
# mapfile="act_planck_dr5.01_s08s18_AA_f%s_daynight_map_srcfree_T_2.10fwhm_minus_SMICAnoSZ_Healpix8192_alm.fits"%str(f).zfill(3)	###New Smoothed ACT minus SMICAnoSZ in Healpix8192 alm
# savename="actplanck5_f%s_srcfree_T_2.10fwhm_minus_SMICAnoSZ_Healpix8192_eBOSS_ELG_SGC-vDR16"%(str(f).zfill(3))

# if os.path.isfile(cname) and os.path.isfile(mapfile):
# 	Healpix_stack.Stack_radial_avg_interpolate(mapfile,cname,new_rad_avg,res,savename,alm_T_map_F=True)
# else:
# 	print('No catalog or map file by the name: ', cname, mapfile)

###eBOSS New NGC-ACToverlap ELG catalog#########
# f=Freq[n%3]
# print(f)

# ###Healpix alm
# cname="./Whole_Catalogs/eBOSS_ELG_NGC-ACToverlap-vDR16.pkl"		###465199 galaxies in ACT Field, no overlap with SPT
# mapfile="act_planck_dr5.01_s08s18_AA_f%s_daynight_map_srcfree_T_2.10fwhm_minus_SMICAnoSZ_Healpix8192_alm.fits"%str(f).zfill(3)	###New Smoothed ACT minus SMICAnoSZ in Healpix8192 alm
# savename="actplanck5_f%s_srcfree_T_2.10fwhm_minus_SMICAnoSZ_Healpix8192_eBOSS_ELG_NGC-ACToverlap-vDR16"%(str(f).zfill(3))

# if os.path.isfile(cname) and os.path.isfile(mapfile):
# 	Healpix_stack.Stack_radial_avg_interpolate(mapfile,cname,new_rad_avg,res,savename,alm_T_map_F=True)
# else:
# 	print('No catalog or map file by the name: ', cname, mapfile)

###eBOSS New SGC RANDOM ELG catalog#########
f=Freq[n%3]
print(f)
div=24
d=n//3+1
###Healpix alm
cname="./Whole_Catalogs/eBOSS_ELG_SGC-random-vDR16_div%iof%i.pkl"%(d,div)
mapfile="act_planck_dr5.01_s08s18_AA_f%s_daynight_map_srcfree_T_2.10fwhm_minus_SMICAnoSZ_Healpix8192_alm.fits"%str(f).zfill(3)	###New Smoothed ACT minus SMICAnoSZ in Healpix8192 alm
savename="actplanck5_f%s_srcfree_T_2.10fwhm_minus_SMICAnoSZ_Healpix8192_eBOSS_ELG_SGC-random-vDR16_div%iof%i"%(str(f).zfill(3),d,div)

if os.path.isfile(cname) and os.path.isfile(mapfile):
	Healpix_stack.Stack_radial_avg_interpolate(mapfile,cname,new_rad_avg,res,savename,alm_T_map_F=True)
else:
	print('No catalog or map file by the name: ', cname, mapfile)



###eBOSS griW ELG catalog#########
# f=Freq[n%3]
# print(f)

# ###Healpix alm
# cname="./Whole_Catalogs/eBOSS_ELG_griW_ACT.pkl"		###465199 galaxies in ACT Field, no overlap with SPT
# mapfile="act_planck_dr5.01_s08s18_AA_f%s_daynight_map_srcfree_T_2.10fwhm_minus_SMICAnoSZ_Healpix8192_alm.fits"%str(f).zfill(3)	###New Smoothed ACT minus SMICAnoSZ in Healpix8192 alm
# savename="actplanck5_f%s_srcfree_T_2.10fwhm_minus_SMICAnoSZ_Healpix8192_eBOSS_ELG_griW_ACT"%(str(f).zfill(3))

# if os.path.isfile(cname) and os.path.isfile(mapfile):
# 	Healpix_stack.Stack_radial_avg_interpolate(mapfile,cname,new_rad_avg,res,savename,alm_T_map_F=True)
# else:
# 	print('No catalog or map file by the name: ', cname, mapfile)

###eBOSS UgrizW ELG catalog#########
# f=Freq[n%3]
# print(f)

# ###Healpix alm
# cname="./Whole_Catalogs/eBOSS_ELG_UgrizW_ACT.pkl"		###465199 galaxies in ACT Field, no overlap with SPT
# mapfile="act_planck_dr5.01_s08s18_AA_f%s_daynight_map_srcfree_T_2.10fwhm_minus_SMICAnoSZ_Healpix8192_alm.fits"%str(f).zfill(3)	###New Smoothed ACT minus SMICAnoSZ in Healpix8192 alm
# savename="actplanck5_f%s_srcfree_T_2.10fwhm_minus_SMICAnoSZ_Healpix8192_eBOSS_ELG_UgrizW_ACT"%(str(f).zfill(3))

# if os.path.isfile(cname) and os.path.isfile(mapfile):
# 	Healpix_stack.Stack_radial_avg_interpolate(mapfile,cname,new_rad_avg,res,savename,alm_T_map_F=True)
# else:
# 	print('No catalog or map file by the name: ', cname, mapfile)


###eBOSS UgrizW ELG Random Radial###
# f=Freq[n%3]
# divs=12
# d=n//3+1
# print(f,d)

# ###Healpix alm
# cname="./Whole_Catalogs/eBOSS_ELG_UgrizW_ACT_rand_1mil_div%iof%i.pkl"%(d,divs)		###465199 galaxies in ACT Field, no overlap with SPT
# mapfile="act_planck_dr5.01_s08s18_AA_f%s_daynight_map_srcfree_T_2.10fwhm_minus_SMICAnoSZ_Healpix8192_alm.fits"%str(f).zfill(3)	###New Smoothed ACT minus SMICAnoSZ in Healpix8192 alm
# savename="actplanck5_f%s_srcfree_T_2.10fwhm_minus_SMICAnoSZ_Healpix8192_eBOSS_ELG_UgrizW_ACT_rand_1mil_div%iof%i"%(str(f).zfill(3),d,divs)

# if os.path.isfile(cname) and os.path.isfile(mapfile):
# 	Healpix_stack.Stack_radial_avg_interpolate(mapfile,cname,new_rad_avg,res,savename,alm_T_map_F=True)
# else:
# 	print('No catalog or map file by the name: ', cname, mapfile)


###Single Catalog Stacking (submitting jobs per frequency)
# # mapfile='act_planck_dr5.01_s08s18_AA_f%s_dn_map_act-spt-overlap.fits'%str(Freq[n%3]).zfill(3)
# mapfile="act_planck_dr5.01_s08s18_AA_f%s_daynight_map_srcfree_T_2.10fwhm_minus_SMICAnoSZ.fits"%str(Freq[n%3]).zfill(3)	###New Smoothed minus SMICAnoSZ
# # cname='./Whole_Catalogs/SPT-ACT_Random699389_div10_%i.txt'%(n//3)						###Radom Catalog (cut up into 10 divisions)
# # savename='ACT_dn_SPT-ACT_Random699389_div10_%i_f%s'%(n//3,str(Freq[n%3]).zfill(3))

# # cname='./Whole_Catalogs/des_wise_spt-act-overlap_select_wclusters_v00.pkl'		###Galaxy Catalog
# # savename='ACT_dn_des_wise_wclusters_f%s_spt-act-overlap'%(str(Freq[n]).zfill(3))

# # cname='./Whole_Catalogs/des_wise_spt-act-overlap_30arc_edges_select_wclusters_v00.pkl'		###Galaxy Catalog
# # savename='ACT_dn_des_wise_wclusters_f%s_spt-act-overlap_30arc_edges'%(str(Freq[n]).zfill(3))

# # cname='./Whole_Catalogs/des_wise_spt-act-overlap_95-10-outliers_select_wclusters_v00.pkl'		###Galaxy Catalog
# # savename='ACT_dn_des_wise_wclusters_f%s_spt-act-overlap_95-10-outliers'%(str(Freq[n]).zfill(3))

# # cname="Whole_Catalogs/des_wise_spt-act-overlap_10arc105043_select_wclusters_v00_v1.pkl"
# # savename='ACT_dn_des_wise_wclusters_f%s_spt-act-overlap_10arc105043_select_wclusters_v00_v1'%(str(Freq[n]).zfill(3))

# cname="Whole_Catalogs/des_wise_spt-act-overlap_20arc94452_select_wclusters_v00_v1.pkl"
# savename='ACT_dn_srcfree_2.10fwhm_minus_SMICAnoSZ_CAR_des_wise_wclusters_f%s_spt-act-overlap_20arc94452_select_wclusters_v00_v1'%(str(Freq[n]).zfill(3))
# if os.path.isfile(cname) and os.path.isfile(mapfile):
# 	# ACT_stack.Stack(mapfile=mapfile,catfile=cname,side_arcmin=arc,res=res,datasavename=savename)
# 	ACT_stack.Stack_interpolate(mapfile=mapfile,catfile=cname,side_arcmin=arc,res=res,datasavename=savename)
# 	# ACT_stack.CAP_Stack(mapfile=mapfile,catfile=cname,CAP_R_set=CAP_R_set,res=res,datasavename=savename)
# else:
# 	print('No catalog or map file by the name: ', cname, mapfile)


# ###Healpix alm version (CHECK)
# mapfile="act_planck_dr5.01_s08s18_AA_f%s_daynight_map_srcfree_T_2.10fwhm_minus_SMICAnoSZ_Healpix8192_alm.fits"%str(Freq[n%3]).zfill(3)	###New Smoothed ACT minus SMICAnoSZ in Healpix8192 alm
# cname="Whole_Catalogs/des_wise_spt-act-overlap_20arc94452_select_wclusters_v00_v1.pkl"
# savename='ACT_dn_srcfree_2.10fwhm_minus_SMICAnoSZ_Healpix8192_des_wise_wclusters_f%s_spt-act-overlap_20arc94452_select_wclusters_v00_v1'%(str(Freq[n]).zfill(3))
# if os.path.isfile(cname) and os.path.isfile(mapfile):
# 	# Healpix_stack.Stack(mapfile=mapfile,catfile=cname,side_arcmin=arc,res=res,datasavename=savename,alm_T_map_F=True)
# 	Healpix_stack.Stack_interpolate(mapfile=mapfile,catfile=cname,side_arcmin=arc,res=res,datasavename=savename,alm_T_map_F=True)
# else:
# 	print('No catalog or map file by the name: ', cname, mapfile)

###z-binned Catalog Stacking (submitting jobs per frequency)
# lowz_cname='./Whole_Catalogs/des_wise_act_select_wclusters_v00_lowz.pkl'		###Galaxy Catalog low-z (z<1.1)
# highz_cname='./Whole_Catalogs/des_wise_act_select_wclusters_v00_highz.pkl'		###Galaxy Catalog low-z (z>=1.1)
# cname_set=[lowz_cname,highz_cname]
# mapfile='act_planck_dr5.01_s08s18_AA_f%s_dn_map_act-spt-overlap.fits'%str(Freq[n%3]).zfill(3)
# lowz_savename='actplanck5_dn_des_wise_wclusters_f%s_lowz'%(str(Freq[n%3]).zfill(3))
# highz_savename='actplanck5_dn_des_wise_wclusters_f%s_highz'%(str(Freq[n%3]).zfill(3))
# savename_set=[lowz_savename,highz_savename]

# cname=cname_set[n//3]
# savename=savename_set[n//3]
# if os.path.isfile(cname) and os.path.isfile(mapfile):
# 	ACT_stack.Stack(mapfile=mapfile,catfile=cname,side_arcmin=arc,res=res,datasavename=savename)
# else:
# 	print('No catalog or map file by the name: ', cname, mapfile)

####Mass binning version (stacking single frequency with mass bins task id)
# binsize=0.1
# # mbins=np.arange(10.95,12.01,binsize)    ###The original 
# mbins=np.arange(10.95,12.11,binsize)
# print(mbins)
# f=Freq[n%3]
# m=np.round(mbins[n//3],2)
# print(m,f)
# # cname='./Whole_Catalogs/act_des_highz_4arc_wclusters_mbin%.2f_%.1fbinsize.pkl'%(m,binsize)			####4arc
# # savename='actplanck5_f%s_T_des_highz_4arc_wclusters_m%.2f_%.1fBin'%(str(f).zfill(3),m,binsize)
# cname='./Whole_Catalogs/act_des_highz_wclusters_mbin%.2f_%.1fbinsize.pkl'%(m,binsize)			###No 4arc
# savename='actplanck5_f%s_T_des_highz_wclusters_m%.2f_%.1fBin'%(str(f).zfill(3),m,binsize)
# mapfile='act_planck_dr5.01_s08s18_AA_f%s_dn_map_act_T.fits'%str(f).zfill(3)
# if os.path.isfile(cname) and os.path.isfile(mapfile):
# 	ACT_stack.Stack(mapfile=mapfile,catfile=cname,side_arcmin=arc,res=res,datasavename=savename)
# else:
# 	print('No catalog file by the name: ', cname)


####Mass binning ---RMS--- version (stacking single frequency with mass bins task id)					RMS
# binsize=0.1
# # mbins=np.arange(10.95,12.01,binsize)    ###The original 
# mbins=np.arange(10.95,12.11,binsize)
# print(mbins)
# f=Freq[n%3]
# m=np.round(mbins[n//3],2)
# print(m,f)
# # cname='./Whole_Catalogs/act_des_highz_4arc_wclusters_mbin%.2f_%.1fbinsize.pkl'%(m,binsize)	###4arc version
# # savename='actplanck5_f%s_T_des_highz_4arc_wclusters_m%.2f_%.1fBin'%(str(f).zfill(3),m,binsize)
# cname='./Whole_Catalogs/act_des_highz_wclusters_mbin%.2f_%.1fbinsize.pkl'%(m,binsize)
# savename='actplanck5_f%s_T_des_highz_wclusters_m%.2f_%.1fBin'%(str(f).zfill(3),m,binsize)
# mapfile='act_planck_dr5.01_s08s18_AA_f%s_dn_map_act_T.fits'%str(f).zfill(3)
# if os.path.isfile(cname) and os.path.isfile(mapfile):
# 	ACT_stack.Stack_rms_only(mapfile,cname,arc_set,res,savename)
# else:
# 	print('No catalog file by the name: ', cname)



###For Random location stacking (in 12 divisions to be added together, also separate frequencies)
# div=12
# f=Freq[n%3]
# d=n//3+1
# print(d,f)
# arc=60
# res=0.05
# cname='./Whole_Catalogs/des_wise_rand_act_4arc_1075791_v0_div%iof12.pkl'%(d)		###4arc name
# savename='actplanck5_f%s_T_des_rand_act_1075791_v0_div%iof12'%(str(f).zfill(3),d)
# cname='./Whole_Catalogs/des_wise_rand_act_1mil_v0_div%iof12.pkl'%(d)				###No 4arc
# savename='actplanck5_f%s_T_des_rand_act_1mil_v0_div%iof12'%(str(f).zfill(3),d)
# mapfile='act_planck_dr5.01_s08s18_AA_f%s_dn_map_act_T.fits'%str(f).zfill(3)
# if os.path.isfile(cname) and os.path.isfile(mapfile):
# 	ACT_stack.Stack(mapfile=mapfile,catfile=cname,side_arcmin=arc,res=res,datasavename=savename)
# else:
# 	print('No catalog file by the name: ', cname)

###For Random ---RMS--- stacking stuff
# div=12
# f=Freq[n%3]
# d=n//3+1
# print(d,f)
# # cname='./Whole_Catalogs/des_wise_rand_act_4arc_1075791_v0_div%iof12.pkl'%(d)		###4arc name
# # savename='actplanck5_f%s_T_des_rand_act_1075791_v0_div%iof12'%(str(f).zfill(3),d)
# cname='./Whole_Catalogs/des_wise_rand_act_1mil_v0_div%iof12.pkl'%(d)
# savename='actplanck5_f%s_T_des_rand_act_1mil_v0_div%iof12'%(str(f).zfill(3),d)
# mapfile='act_planck_dr5.01_s08s18_AA_f%s_dn_map_act_T.fits'%str(f).zfill(3)
# if os.path.isfile(cname) and os.path.isfile(mapfile):
# 	ACT_stack.Stack_rms_only(mapfile,cname,arc_set,res,savename)
# else:
# 	print('No catalog file by the name: ', cname)