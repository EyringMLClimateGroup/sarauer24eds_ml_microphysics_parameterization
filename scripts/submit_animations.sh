#!/bin/bash
#SBATCH --partition=compute
#SBATCH --account=bd1179
#SBATCH --nodes=1
#SBATCH --time=01:00:00

srun python ../plotscripts/animation_plot_map.py --variable tend_ta_mig --datatype output
srun python ../plotscripts/animation_plot_map.py --variable tend_qv_mig --datatype output
srun python ../plotscripts/animation_plot_map.py --variable tend_qcli_mig --datatype output
srun python ../plotscripts/animation_plot_map.py --variable tend_qclw_mig --datatype output
srun python ../plotscripts/animation_plot_map.py --variable tend_qg_mig --datatype output
srun python ../plotscripts/animation_plot_map.py --variable tend_qs_mig --datatype output
srun python ../plotscripts/animation_plot_map.py --variable tend_qr_mig --datatype output


#srun python ../plotscripts/animation_plot_map.py --variable ta_mig --datatype input
#srun python ../plotscripts/animation_plot_map.py --variable qv_mig --datatype input
#srun python ../plotscripts/animation_plot_map.py --variable qi_mig --datatype input
#srun python ../plotscripts/animation_plot_map.py --variable qc_mig --datatype input
#srun python ../plotscripts/animation_plot_map.py --variable qg_mig --datatype input
#srun python ../plotscripts/animation_plot_map.py --variable qs_mig --datatype input
#srun python ../plotscripts/animation_plot_map.py --variable qr_mig --datatype input