 #!/bin/bash
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem-per-cpu=1G
#SBATCH --time=0-03:25:00     # 15 minutes
 
# Put commands for executing job below this line
# This example is loading Python and then
# writing out the version of Python
echo "Beginning executuion"
python_imp=~/miniconda3/envs/imp/bin/python

#arguments are job_type no_repeats and save_directory
mkdir $3
$python_imp compute_data.py $1 $2 $3
$python_imp --version
