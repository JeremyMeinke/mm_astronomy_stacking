#!/bin/sh

###This is an example of a HPC job ticket for submission to the ASU AGAVE computing cluster
###See https://asurc.atlassian.net/wiki/spaces/RC/overview for more info

#SBATCH -n 1
#SBATCH -t 3-00:00                  # wall time (D-HH:MM)
#SBATCH --cpus-per-task=3			###Reduce this if possible	(3 for healpix alms, ~5 for pixell read_map...)
#SBATCH --array=0-3
##SBATCH --dependency=afterok:jobnumber	###If needing to run after another job... very niche purpose.

### Emailing the user about the job(s), kinda my default layer of extra info if debugging something
#SBATCH --mail-type=ALL             # Send a notification when the job starts, stops, or fails.
#SBATCH --mail-user=yourname@email # send-to address

module purge
module load anaconda/py3
source activate skymaps-py	###my custom anaconda environment, see included skymaps-py.yml for dependencies
python hpc_stack_example.py  ${SLURM_ARRAY_TASK_ID}	###Passes array number through function to run multiple jobs parallel (useful for large catalogs)
