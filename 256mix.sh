#!/bin/bash
# 256mix.sh - HPL on 240 A100 + 16 H100 = 256 GPUs, 16x16 grid
# Slurm heterogeneous job: component 0 = A100 nodes (4 GPUs each), later components = H100 nodes.
# H100 nodes: scg006 and scg020 are 8-way, scg016 is 4-way (confirmed 2026-10-14). Check them
# with h100_mix_check.sh. Options after each '#SBATCH hetjob' line belong to the next component.
# Usage:  sbatch 256mix.sh            BENCH=hpl-mxp sbatch 256mix.sh          MIX_NVSHMEM=0 sbatch 256mix.sh
#SBATCH -J hpl_256mix
#SBATCH -o mix256.log
#SBATCH --open-mode=append
#SBATCH --export=NONE
#SBATCH -t 0-4
#SBATCH -p hpl-256
#SBATCH -q public
#SBATCH --reservation=oct2026_maint
#SBATCH -N 60-60
#SBATCH -w sg[001-050],sdg051,sg[235-239],scg[009-012]
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=single:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=512000M
#SBATCH --distribution=block:block
#SBATCH hetjob
#SBATCH -t 0-4
#SBATCH -p hpl-256
#SBATCH -q public
#SBATCH --reservation=oct2026_maint
#SBATCH -N 2-2
#SBATCH -w scg006,scg020
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gpus-per-node=8
#SBATCH --gpu-bind=single:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256000M
#SBATCH --distribution=block:block
DAT_HPL=./dats/256GPU_mix.dat
DAT_MXP=./dats/HPL-MxP_256GPU.dat
DAT_HPCG=./dats/HPCG_256GPU.dat
source "${SLURM_SUBMIT_DIR:-.}/mix_common.sh"
