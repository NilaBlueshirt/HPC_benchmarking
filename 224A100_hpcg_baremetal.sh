#!/bin/bash
# Bare-metal HPCG run for 224 GPUs.
# Adapted from 224A100_baremetal.sh; same modules, env stack, and launcher
# pattern — just points at the HPCG binary inside the v26.02 redistributable.
# NVSHMEM is ON (bare-metal dodges the container's NVSHMEM bootstrap bug).

#SBATCH -J HPCG-bm-224
#SBATCH --reservation=maint
#SBATCH -p hpl
#SBATCH -q public
#SBATCH -t 0-1
#SBATCH -C a100_80
#SBATCH -N 56-56
#SBATCH -w sg[001-050],sdg051,sg[235-239],scg0[09-12]
#SBATCH -x sg034,scg009
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --cpus-per-task=12
#SBATCH --mem=480G
#SBATCH -o hpcg-baremetal.log
#SBATCH --open-mode=append
#SBATCH --distribution=block:block

echo "==============================================================================="
echo "---------------- HPCG Bare-metal Run Start $(date) ------------------"
echo ""

# ---- Load modules ----------------------------------------------------------
module purge
module load nvhpc/26.3-cuda-13.1-nompi
module load hpcx/2.25.1

# ---- HPCG package location -------------------------------------------------
# Same v26.02 extraction as HPL; just a different sub-dir.
# Expected layout:
#   cuda13/hpcg-linux-x86_64/hpcg.sh
#   cuda13/hpcg-linux-x86_64/xhpcg
#   cuda13/hpcg-linux-x86_64/sample-dat/
HPLDIR=${HPLDIR:-$HOME/HPL/nv_binary/nvidia_hpc_benchmarks_openmpi-linux-x86_64-26.02.02-archive/cuda13}
HPCGBIN=${HPCGBIN:-${HPLDIR}/hpcg-linux-x86_64}

echo "=== Loaded modules ==="
module list 2>&1
echo ""
echo "=== HPCG package check ==="
echo "HPCGBIN=${HPCGBIN}"
ls -la "${HPCGBIN}/hpcg.sh" 2>&1 || { echo "ERROR: hpcg.sh not at ${HPCGBIN}/hpcg.sh — adjust HPCGBIN. Try: ls ${HPLDIR}"; exit 1; }
# Binary may be named xhpcg or xhpcg-3d in different versions; check both:
ls -la "${HPCGBIN}/xhpcg"* 2>&1 || { echo "ERROR: xhpcg* binary missing under ${HPCGBIN}"; exit 1; }
echo ""

# ---- DAT file --------------------------------------------------------------
DAT_FILE=${DAT_FILE:-$(readlink -f ./dats/HPCG-224GPU.dat)}

echo "SBATCH script for JOB ${SLURM_JOB_ID} :"
scontrol write batch_script $SLURM_JOB_ID - | grep -v '^ *echo '
echo ""
echo "Using DAT File: ${DAT_FILE}"
cat ${DAT_FILE} 2>&1 || { echo "ERROR: DAT file missing at ${DAT_FILE}"; exit 1; }
echo ""

# ---- Per-rank wrapper ------------------------------------------------------
mkdir -p myhpl
cat > myhpl/run-hpcg-baremetal.sh <<EOF
#!/bin/bash
ulimit -l unlimited 2>/dev/null

# Re-load modules in this child shell — slurmstepd doesn't always pass through
# the parent's module-set env vars. Same fix that unblocked HPL-MxP.
source /etc/profile.d/lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null
module purge >/dev/null 2>&1
module load nvhpc/26.3-cuda-13.1-nompi
module load hpcx/2.25.1

# Ensure UCX libs are findable for OpenMPI's mca_pml_ucx.so.
# hpl.sh adds the UCX path internally; hpcg.sh and hpl-mxp.sh may not.
HPCX_INIT=/packages/apps/hpcx/2.25.1/inbox/hpcx-init.sh
[ -f "\$HPCX_INIT" ] && source "\$HPCX_INIT" 2>/dev/null && (hpcx_load 2>/dev/null || true)
UCX_LIB=\$(find /packages/apps/hpcx/2.25.1 -name "libucp.so*" 2>/dev/null | head -1 | xargs -r dirname)
[ -n "\$UCX_LIB" ] && export LD_LIBRARY_PATH=\$UCX_LIB:\$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/packages/apps/hpcx/2.25.1/inbox/ucx/lib:\$LD_LIBRARY_PATH

# First-rank diagnostic — confirm UCX/PML can load
if [ "\$SLURM_PROCID" = "0" ]; then
  echo "DEBUG (rank 0): which mpirun = \$(which mpirun 2>&1)"
  echo "DEBUG (rank 0): UCX_LIB = \$UCX_LIB"
  echo "DEBUG (rank 0): pml plugins available:"
  ompi_info --param pml all 2>&1 | grep "MCA pml:" | head -5
fi

HCA=\$(basename /sys/class/net/ib0/device/infiniband/* 2>/dev/null || ibdev2netdev | awk "/ ib0 /{print \\\$1; exit}")
MAP=\${HCA}:1
GPU_N=\${SLURM_GPUS_ON_NODE:-\$(nvidia-smi -L 2>/dev/null | wc -l || echo 4)}
PE_MAP=\$(printf "%s," \$(yes "\$MAP" | head -n "\$GPU_N")); PE_MAP=\${PE_MAP%,}

echo "Host=\$(hostname) SLURM_PROCID=\$SLURM_PROCID SLURM_LOCALID=\$SLURM_LOCALID CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES PE_MAP=\$PE_MAP"

# Threading
export OMP_NUM_THREADS=12
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# OpenMPI / UCX
export OMPI_MCA_pml=ucx
export OMPI_MCA_coll=^ucc
export OMPI_MCA_btl=^openib
export PMIX_MCA_gds=hash
export UCX_TLS=dc_x,sm,self,cuda_ipc,cuda_copy,gdr_copy
export UCX_NET_DEVICES=\$MAP
export UCX_SOCKADDR_TLS_PRIORITY=rdmacm
export UCX_POSIX_USE_PROC_LINK=n
export UCX_IB_GDR_DIRECT_RDMA=y

# NVSHMEM — ON (bare-metal works for HPL; expect same for HPCG)
# Set both HPL_* and HPCG_* env var families for safety (whichever the
# launcher reads will be honored; the other is just ignored).
export HPL_USE_NVSHMEM=1
export HPL_NVSHMEM_SWAP=1
export HPL_NVSHMEM_INIT=0
export HPCG_USE_NVSHMEM=1
export HPCG_NVSHMEM_INIT=0
export HPL_ALLOC_HUGEPAGES=1
export NVSHMEM_REMOTE_TRANSPORT=ibrc
export NVSHMEM_HCA_PE_MAPPING=\$PE_MAP
export NVSHMEM_FORCE_IBGDA=1
export NVSHMEM_DEBUG=WARN

# NCCL
export NCCL_DEBUG=WARN
export NCCL_NET=IB
export NCCL_COLLNET_ENABLE=1
export NCCL_IB_HCA=\$HCA
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_GID_INDEX=0
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_CROSS_NIC=0
# Intentionally NOT setting NCCL_P2P_DISABLE — want intra-node NVLink P2P.

# Monitoring
export MONITOR_GPU=1
export GPU_TEMP_WARNING=80
export GPU_CLOCK_WARNING=562

# Run HPCG via the package's launcher.
# --dat passes the input file with local-grid + runtime settings.
# --gpu-affinity and --ucx-affinity are per-local-rank (skipping cpu/mem-affinity
# because NUMA topology differs across some nodes — same as HPL-MxP).
${HPCGBIN}/hpcg.sh --nx 384 --ny 384 --nz 384 --rt 1800 \\
    --gpu-affinity 0:1:2:3 \\
    --ucx-affinity mlx5_2:mlx5_2:mlx5_2:mlx5_2
EOF
chmod +x myhpl/run-hpcg-baremetal.sh

# ---- Launch ----------------------------------------------------------------
srun --mpi=pmix --export=ALL -n $SLURM_NTASKS \
    ./myhpl/run-hpcg-baremetal.sh

echo "---------------- HPCG Bare-metal Run End $(date) ------------------"
echo "==============================================================================="
