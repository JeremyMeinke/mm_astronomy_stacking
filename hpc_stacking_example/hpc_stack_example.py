from __future__ import division
import numpy as np
import sys
import os.path
from astropy.table import Table
import stacking	###Same folder now (if moved stacking.py and these hpc files to same HPC folder)

###Master Variables for whatever used further below
n = int(sys.argv[1])
spt_freq = [95,150,220]
side_arcmin = 60		#[arcmin] stamp side size, aka arc x arc dimension image
pix_res = 0.05			#Default size for act pixell thumbnails (and coincidentally my chosen resolution for SPT stamps... so yay)

###If radially averaging
rad_step = 0.50
rad_avg = np.arange(0,20.1,rad_step)

###Assuming all files are saved in the same location on the HPC server account:
catalog_name = "2500d_cluster_sample_Bocquet19.fits"
map_file_name = "combined_map_%iGHz_nside8192_ptsrcmasked_50mJy.fits"%spt_freq[n]
save_name = "SPT_combined_map_%iGHz_ptsrcmasked_Bocquet19_clusters"%spt_freq[n]

if os.path.isfile(catalog_name):
	###Catalog Loading and getting RAs and DECs.  This may differ greating according to catalog and file types
	catalog = Table.read("./data/2500d_cluster_sample_Bocquet19.fits", format="fits").to_pandas()
	catalog_ra = catalog["RA"]
	catalog_dec = catalog["DEC"]
	###Now the map, also can differ according to map type (healpix vs CAR projection, etc.)
	if os.path.isfile(map_file_name):
		spt_map = stacking.map_data_healpix_fits(map_file_name, unreadable_header=True)
		###Note the data_save_name is now given, which will save a stack as a .txt file with _%.2farcmin_%.2fpixres.txt attached
		stacking.stack(spt_map, catalog_ra.to_numpy(), catalog_dec.to_numpy(), side_arcmin, pix_res, data_save_name=save_name, interpolate=True)
		
		###Alternatively for radial average measurements, which will save as a .csv pandas dataframe/table with _radial_avg_%.2fpixres.csv
		# stacking.stack_radial_avg(spt_map, catalog_ra.to_numpy(), catalog_dec.to_numpy(), rad_avg, pix_res, data_save_name=save_name)


	else:	###Just a simply way to verify any quick mistakes
		print("Map file name incorrect or not present...")
else:
	print("Catalog file name incorrect or not present...")

