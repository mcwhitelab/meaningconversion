#!/bin/sh
# -------------
#SBATCH --job-name=run_job
#SBATCH --account=mcwhite
#SBATCH --partition=gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL

source ~/.bashrc
conda activate /groups/clairemcwhite/envs/core_pkgs
module load clustal-omega/1.2.4


f=/groups/clairemcwhite/claire_workspace/github/meaningconversion/fastas/msp.fasta

python /groups/clairemcwhite/claire_workspace/github/meaningconversion/scripts/seqtransform_az.py -m /groups/clairemcwhite/models/prot_t5_xl_uniref50/ -f $f --optimization-method beam --look-ahead 5 --max_steps 30 -e 5

