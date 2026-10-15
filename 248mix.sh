#!/bin/bash
# 248mix.sh - HPL on 240 A100 + 8 H100 (4 GPUs on each of scg006 and scg020) = 248 GPUs, 8x31 grid.
# One plain job with 4 tasks and 4 GPUs on every node. NVSHMEM requires the same number of PEs on
# all nodes, which is why the 4+8 layout of 256mix.sh cannot run (2026-10-14). Slurm picks 4 of the
# 8 GPUs on the 8-way nodes; the "On node" lines in the log show which ones (PCI bus id).
# Usage:  sbatch 248mix.sh            sbatch 248mix.sh DAT=dats/x.dat            sbatch 248mix.sh BENCH=hpcg
#SBATCH -J hpl_248mix
#SBATCH -o mix248.log
#SBATCH --open-mode=append
#SBATCH --export=NONE
#SBATCH -t 0-4
#SBATCH -p hpl-256
#SBATCH -q public
#SBATCH --reservation=oct2026_maint
#SBATCH -N 62-62
#SBATCH -w sg[001-050],sdg051,sg[235-239],scg[006,009-012,020]
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=single:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --distribution=block:block
for a in "$@"; do case "$a" in [A-Za-z_]*=*) export "$a" ;; esac; done   # knobs come as arguments: the header's --export=NONE drops VAR=x prefixes
DAT_HPL=./dats/248GPU_mix.dat
DAT_MXP=./dats/HPL-MxP_248GPU.dat    # not written: HPL-MxP would need its own 8x31 input
DAT_HPCG=./dats/HPCG_256GPU.dat      # HPCG takes any rank count
source "${SLURM_SUBMIT_DIR:-.}/mix_common.sh"
