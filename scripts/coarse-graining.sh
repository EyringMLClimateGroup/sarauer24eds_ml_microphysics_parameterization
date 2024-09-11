#!/bin/bash

# mistral cpu batch job parameters
# --------------------------------
#SBATCH --account=bd1179
#SBATCH --job-name=coarse-grain-run_20200125T000000-20200129T235920
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --threads-per-core=1
# the following is needed to work around a bug that otherwise leads to
# a too low number of ranks when using compute,compute2 as queue
#SBATCH --mem=0
#SBATCH --output=/work/bd1179/b309246/log_coarse-grain/coarse-grain-run_20200125T000000-20200129T235920.log
#SBATCH --time=10:00:00

#=============================================================================
# Module loading
#=============================================================================
module load gcc/11.2.0-gcc-11.2.0
module load openmpi/4.1.2-gcc-11.2.0
module load cdo/2.0.4-gcc-11.2.0

# -------------------------------------------------------------------------------
#                            EXPERIMENT SETTINGS
# -------------------------------------------------------------------------------
# --------------------------------------
## TODO expname
## Use the same as in the ICON runscript
## No "exp." and no ".run"
#file="r2b9_amip_atm_mig_tendencies_ml_20200125T000000Z.nc"
expname="run_20200130T000000-20200203T235920"
## TODO paths
## Edit the paths if you save data in /scratch, /project, ...
inpath="/work/bd1179/b309246/experiments/r2b9_amip/${expname}/"
outpath="/scratch/b/b309246/coarse-grained-data/"

target_grid="/pool/data/ICON/grids/public/mpim/0019/icon_grid_0019_R02B05_G.nc"
weights="/work/bd1179/b309246/grids/weights_R2B5_from_R2B9.nc"
source_grid="/work/bd1179/b309246/grids/cell_area_0015_R02B09_G.nc"

foo () {
    echo $file
    # Read filename
    file_name=`echo $file | cut -d "." -f1`
    file=`echo $file`
    # Do not overwrite (there are 42 files per variable to process)
    if [[ $file_name =~ ""("2d_general_vars"|"atm_cl"|"atm_mig_inputs"|"atm_mig_tendencies"|"atm_mon"|"r2b9_amip_lnd")"" ]];then
    #if [[ $file_name =~ ""("atm_mig_inputs_ml_20200128T000000Z"|"atm_mig_inputs_ml_20200129T000000Z"|"atm_mig_inputs_ml_20200130T000000Z")"" ]];then

        echo $file_name
        if ! [[ -f ${outpath}/${file_name}.nc ]];then
            #if [[ $file =~ ""("20040401T000000Z.grb"|"20040401T030000Z.grb"|"20040401T013600Z.grb")"" ]];then
            # Add resolution to the end of the filename
            # Create placeholder file as quickly as possible (to avoid that two processes are writing the same file)
            touch ${outpath}/${file_name}_R02B05.nc
            echo ${file_name}_R02B05.nc
             
            # I think we can run 36 processes per compute node
            cdo -f nc -P 36 remap,${target_grid},${weights} ${inpath}/${file} ${outpath}/${file_name}_R02B05.nc
            #cdo -f nc copy ${inpath}/${file} ${outpath}/${file_name}.nc
            #fi
        fi
    fi
}



files="`ls $inpath`"
# No more than N processes run at the same time
N=6

#open_sem $base $N
for file in $files; do
    foo $file
done 

# Wait until all threads are finished
wait