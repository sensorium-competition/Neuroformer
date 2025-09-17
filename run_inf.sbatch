#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=react    # see https://www.hlrn.de/doc/display/PUB/Compute+node+partitions for a list of accessible hlrn partitions
#SBATCH -G A100:1                   # number of gpus per node
#SBATCH --time 0-48:00
#SBATCH --constraint "inet" 
#SBATCH -A agsinz_foundation_model_brain
#SBATCH --mem=100G
#SBATCH --output=./slurm_files/slurm-%x-%j.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=./slurm_files/slurm-%x-%j.err      # where to write slurm error
#SBATCH --mail-type=begin            # send mail when job begins
#SBATCH --mail-type=end              # send mail when job ends
#SBATCH --mail-user=goirik.chakrabarty@uni-goettingen.de
export HTTP_PROXY="http://www-cache.gwdg.de:3128"
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"

module load apptainer
module load cuda/12.2.1
module load miniforge3
module load git

source /sw/tools/python/anaconda3/2020.11/skl/bin/activate 
conda info --envs
conda activate /mnt/vast-react/projects/agsinz_foundation_model_brain/.conda/envs/neuroformer
cd /mnt/vast-react/projects/agsinz_foundation_model_brain/goirik/Neuroformer/

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# For debugging purposes.
python --version
python -m torch.utils.collect_env
#nvcc -V
# python neuroformer_train.py --dataset V1AL --config configs/Visnav/lateral/mconf_pretrain.yaml
# python neuroformer_train_experanto_multisession.py --dataset experanto --config configs/Experanto/mconf_all_30Hz_two.yaml
# python neuroformer_train_experanto_multisession.py --dataset experanto --config configs/Experanto/mconf_all_30Hz_published-5.yaml
# python neuroformer_train_experanto_multisession.py --dataset experanto --config configs/Experanto/mconf_all_30Hz_all-12.yaml
# python neuroformer_inference_multi.py --dataset experanto --config configs/Experanto/mconf_all_30Hz_two.yaml --ckpt_path ./models/test --dataset experanto --predict_modes ['treadmill']
python neuroformer_inference_multi.py --dataset experanto --config configs/Experanto/mconf_all_30Hz_all-12.yaml --ckpt_path ./models/test_7 --dataset experanto --predict_modes treadmill