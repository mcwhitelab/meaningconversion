#!/bin/sh
# -------------
#SBATCH --job-name=run_job
#SBATCH --account=mcwhite
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL

source ~/.bashrc
conda activate /groups/clairemcwhite/envs/core_pkgs
module load clustal-omega/1.2.4

f=/groups/clairemcwhite/claire_workspace/github/meaningconversion/fastas/msp.fasta
m=prot_t5_xl_uniref50/ 
python /groups/clairemcwhite/claire_workspace/github/meaningconversion/scripts/seqtransform_embed.py -m /groups/clairemcwhite/models/$m -f $f --beam-search --look-ahead 5 --max_steps 20 -e 100 --n_subsamples 5 --subsample_size 20 --baseline-reward

m=esm2_t30_150M_UR50D
python /groups/clairemcwhite/claire_workspace/github/meaningconversion/scripts/seqtransform_embed.py -m /groups/clairemcwhite/models/$m -f $f --beam-search --look-ahead 5 --max_steps 20 -e 100 --n_subsamples 5 --subsample_size 20 --baseline-reward



m=prot_t5_xl_uniref50/ 
python /groups/clairemcwhite/claire_workspace/github/meaningconversion/scripts/seqtransform_az.py -m /groups/clairemcwhite/models/$m -f $f --beam-search --look-ahead 5 --max_steps 30 -e 1000 --n_subsamples 100 --subsample_size 100

m=esm2_t30_150M_UR50D
python /groups/clairemcwhite/claire_workspace/github/meaningconversion/scripts/seqtransform_az.py -m /groups/clairemcwhite/models/$m -f $f --beam-search --look-ahead 5 --max_steps 30 -e 1000 --n_subsamples 100 --subsample_size 100

