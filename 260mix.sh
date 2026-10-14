#!/bin/bash
# 260mix.sh - HPL on 240 A100 + 20 H100 = 260 GPUs, 13x20 grid
# Slurm heterogeneous job: component 0 = A100 nodes (4 GPUs each), later components = H100 nodes.
# H100 nodes: scg006 and scg020 are 8-way, scg016 is 4-way (confirmed 2026-10-14). Check them
# with h100_mix_check.sh. Options after each '#SBATCH hetjob' line belong to the next component.
# Usage:  sbatch 260mix.sh            BENCH=hpl-mxp sbatch 260mix.sh          MIX_NVSHMEM=0 sbatch 260mix.sh
#SBATCH -J hpl_260mix
#SBATCH -o mix260.log
#SBATCH --open-mode=append
#SBATCH --export=NONE
#SBATCH -t 0-4
#SBATCH -p hpl
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
#SBATCH -p hpl
#SBATCH -q public
#SBATCH --reservation=oct2026_maint
#SBATCH -N 2-2
#SBATCH -w scg006,scg020
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gpus-per-node=8
#SBATCH --gpu-bind=single:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256000M
#SBATCH --distribution=block:block
#SBATCH hetjob
#SBATCH -t 0-4
#SBATCH -p hpl
#SBATCH -q public
#SBATCH --reservation=oct2026_maint
#SBATCH -N 1-1
#SBATCH -w scg016
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=single:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256000M
#SBATCH --distribution=block:block
DAT_HPL=./dats/260GPU_mix.dat
DAT_MXP=./dats/HPL-MxP_260GPU.dat
DAT_HPCG=./dats/HPCG_260GPU.dat
source "${SLURM_SUBMIT_DIR:-.}/mix_common.sh"
