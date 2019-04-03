#!/bin/sh
#PBS -q u-short
#PBS -l select=1:mpiprocs=1:ompthreads=1
#PBS -W group_list=gk37
#PBS -l walltime=01:00:00
cd $PBS_O_WORKDIR
./etc/profile.d/modules.sh
module load anaconda3/4.3.0 cuda9/9.0.176
export PYTHONUSERBASE=/lustre/gk37/k37004/torch
export PATH="$PATH:/lustre/gk37/k37004/bin"
python ./pyscripts/A0_test.py 3 > ./outputs/reedbush_A0_test.out 2>&1
