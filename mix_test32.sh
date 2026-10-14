#!/bin/bash
# mix_test32.sh - mixed test: 4 A100 nodes + 2 8-way H100 nodes = 32 GPUs, 4x8 grid
# Slurm heterogeneous job: component 0 = A100 nodes (4 GPUs each), later components = H100 nodes.
# H100 nodes: scg006 and scg020 are 8-way, scg016 is 4-way (confirmed 2026-10-14). Check them
# with h100_mix_check.sh. Options after each '#SBATCH hetjob' line belong to the next component.
# Usage:  sbatch mix_test32.sh            BENCH=hpl-mxp sbatch mix_test32.sh          MIX_NVSHMEM=0 sbatch mix_test32.sh
#SBATCH -J mix_test32
#SBATCH -o mix_test32.log
#SBATCH --open-mode=append
#SBATCH --export=NONE
#SBATCH -t 0-1
#SBATCH -p hpl-256
#SBATCH -q public
#SBATCH --reservation=oct2026_maint
#SBATCH -N 4-4
#SBATCH -w sg[001-004]
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=single:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=512000M
#SBATCH --distribution=block:block
#SBATCH hetjob
#SBATCH -t 0-1
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
DAT_HPL=./dats/32GPU_mix_test.dat
DAT_MXP=./dats/HPL-MxP_256GPU.dat
DAT_HPCG=./dats/HPCG_1node.dat
source "${SLURM_SUBMIT_DIR:-.}/mix_common.sh"
