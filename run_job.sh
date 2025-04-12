#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=25G
#SBATCH --time=7-00:00:00
#SBATCH --mail-user=smukh039@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="TransSub100"
#SBATCH -p cisl
#SBATCH --gres=gpu:2
#SBATCH --wait-all-nodes=1
#SBATCH --output=output_%j-%N.txt

hostname
date
# # Activate Conda
source /home/csgrad/smukh039/miniforge3/etc/profile.d/conda.sh
conda activate cobevt_env
which python


# /home/csgrad/smukh039/miniforge3/envs/auto_env/bin/python /home/csgrad/smukh039/AutoNetworkingRL/AutoNetworkingRL/cohff_opv2v/main_train.py > run_job_output.log 2>&1
# /home/csgrad/smukh039/miniforge3/envs/cobevt_env/bin/python /home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/opencood/tools/train_camera.py --hypes_yaml /home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/opencood/checkpoints_sub100_mantrans/config.yaml --model_dir  /home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/opencood/checkpoints_sub100_mantrans > run_job_checkpoints_sub100_mantrans_req_output.log 2>&1
/home/csgrad/smukh039/miniforge3/envs/cobevt_env/bin/python /home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/opencood/tools/train_camera.py     --hypes_yaml /home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/opencood/checkpoints_trans_sub100/config.yaml     --model_dir /home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/opencood/checkpoints_trans_sub100 > run_job_singlegpu_checkpoints_sub100_req_output.log 2>&1

# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2     /home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/opencood/tools/train_camera.py     --hypes_yaml /home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/opencood/checkpoints_sub100_idtra/config.yaml     --model_dir /home/csgrad/smukh039/AutoNetworkingRL/CoBEVT_AutoNet/opv2v/opencood/checkpoints_sub100_idtra > run_job_idtra_checkpoints_sub100_req_output.log 2>&1
