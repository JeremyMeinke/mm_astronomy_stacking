## hpc_stacking_example
This contains relevant code for stacking catalogs of ra/dec on HPC (high performance computing) clusters
such as *agave* at ASU.  This is useful when requiring high-resolution cutouts or stacking large catalogs.

To run on an HPC cluster such as *agave*: copy stacking.py, catalog and map data files needed, and the code simliar to that in this folder to your HPC account (I usually just did scp and ssh into username@agave.asu.edu).  Run via a shell script, i.e. "sbatch run\_hpc\_stack\_example.sh"

Also, see included skymaps-py.yml for relevant conda environment packages that I had installed.  (Some packages may need/prefer to be updated.)