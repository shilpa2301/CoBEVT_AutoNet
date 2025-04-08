#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=10G
#SBATCH --time=7-00:00:00
#SBATCH --mail-user=smukh039@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="Cobevt_orig"
#SBATCH -p cisl
#SBATCH --wait-all-nodes=1
#SBATCH --output=output_%j-%N.txt

hostname
date
# # Activate Conda
source /home/csgrad/smukh039/miniforge3/etc/profile.d/conda.sh
conda activate cobevt_env
which python

# /home/csgrad/smukh039/miniforge3/envs/auto_env/bin/python /home/csgrad/smukh039/AutoNetworkingRL/AutoNetworkingRL/cohff_opv2v/main_train.py > run_job_output.log 2>&1
/home/csgrad/smukh039/miniforge3/envs/cobevt_env/bin/python /home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/opencood/tools/train_camera.py --hypes_yaml /home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/opencood/checkpoints_orig/config.yaml --model_dir  /home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/opencood/checkpoints_orig > run_job_checkpoints_orig_req_output.log 2>&1
