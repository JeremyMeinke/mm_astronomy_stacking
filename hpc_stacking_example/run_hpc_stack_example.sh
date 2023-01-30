#!/bin/sh

###This is an example of a HPC job ticket for submission to the ASU AGAVE computing cluster
###See https://asurc.atlassian.net/wiki/spaces/RC/overview for more info

#SBATCH -n 1						## Number of nodes
#SBATCH -t 3-00:00                  ## wall time (D-HH:MM)
#SBATCH --cpus-per-task=3			## When using the SPT alm files, it required the memory associated with 3 CPUs (~12 GB), ACT/pixell maps may require more
#SBATCH --array=0-3					## To run multiples, i.e. per frequency or catalog subsets
##SBATCH --dependency=afterok:jobnumber	## If needing to run after another job... very niche purpose.

### Emailing the user about the job(s), kinda my default layer of extra info if debugging something
#SBATCH --mail-type=ALL             ## Send a notification when the job starts, stops, or fails.
#SBATCH --mail-user=yourname@email	## Send-to address

module purge
module load anaconda/py3
source activate skymaps-py	###my custom anaconda environment, see included skymaps-py.yml for dependencies
python hpc_stack_example.py  ${SLURM_ARRAY_TASK_ID}	###Passes array number through function to run multiple jobs parallel (useful for large catalogs)
