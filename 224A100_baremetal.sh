#!/bin/bash
#SBATCH -J HPL-bm-test
#SBATCH --reservation=maint
#SBATCH -p hpl
#SBATCH -q public
#SBATCH -t 0-2
#SBATCH -C a100_80
#SBATCH -N 4-4
#SBATCH -w sg001,sg002,sg003,sg004
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH -o hpl-baremetal.log
#SBATCH --open-mode=append
#SBATCH --distribution=block:block

echo "==============================================================================="
echo "---------------- HPL Bare-metal Run Start $(date) ------------------"
echo ""

# ---- Load modules ----------------------------------------------------------
# nvhpc gives the CUDA toolkit libs; hpcx gives OpenMPI/UCX runtime that
# the v26.02 package's hpl.sh and xhpl link against.
module purge
module load nvhpc/26.3-cuda-13.1-nompi
module load hpcx/2.25.1

# ---- HPL package location --------------------------------------------------
# Bare-metal v26.02 redistributable. Layout under cuda13/:
#   cuda13/hpl-linux-x86_64/hpl.sh         <- launcher script
#   cuda13/hpl-linux-x86_64/xhpl           <- binary
#   cuda13/hpl-linux-x86_64/sample-dat/    <- prebuilt .dat files (HPL-1GPU.dat, HPL-4GPUs.dat, HPL-16GPUs.dat, ...)
# Adjust HPLDIR to wherever you extracted the tarball.
HPLDIR=${HPLDIR:-$HOME/HPL/nv_binary/nvidia_hpc_benchmarks_openmpi-linux-x86_64-26.02.02-archive/cuda13}
HPLBIN=${HPLDIR}/hpl-linux-x86_64

echo "=== Loaded modules ==="
module list 2>&1
echo ""
echo "=== HPL package check ==="
echo "HPLDIR=${HPLDIR}"
echo "HPLBIN=${HPLBIN}"
ls -la "${HPLBIN}/hpl.sh" 2>&1 || { echo "ERROR: hpl.sh not found at ${HPLBIN}/hpl.sh — adjust HPLDIR"; exit 1; }
ls -la "${HPLBIN}/xhpl" 2>&1   || { echo "ERROR: xhpl binary missing under ${HPLBIN}"; exit 1; }
echo ""

DAT_FILE=${DAT_FILE:-${HPLBIN}/sample-dat/HPL-16GPUs.dat}

echo "SBATCH script for JOB ${SLURM_JOB_ID} :"
scontrol write batch_script $SLURM_JOB_ID - | grep -v '^ *echo '
echo ""
echo "Using DAT File: ${DAT_FILE}"
cat ${DAT_FILE} 2>&1 || { echo "ERROR: DAT file missing at ${DAT_FILE}"; exit 1; }
echo ""

# ---- Per-rank wrapper ------------------------------------------------------
# Same env-var structure as the container's run.sh, with three changes:
#   1. NVSHMEM is ON (whole point of bare-metal test)
#   2. NO CUDA_VISIBLE_DEVICES=$SLURM_LOCALID pin (NVSHMEM needs all 4 GPUs visible)
#   3. Uses bare-metal hpl.sh, not container's /workspace/hpl.sh
mkdir -p myhpl
cat > myhpl/run-baremetal.sh <<EOF
#!/bin/bash
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

# NVSHMEM — ON for this test (the whole reason we're going bare-metal)
export HPL_USE_NVSHMEM=1
export HPL_NVSHMEM_SWAP=1
export HPL_NVSHMEM_INIT=0
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
export NCCL_P2P_DISABLE=1

# Monitoring
export MONITOR_GPU=1
export GPU_TEMP_WARNING=80
export GPU_CLOCK_WARNING=562

# Run HPL (bare-metal path — NOT /workspace/hpl.sh)
${HPLBIN}/hpl.sh --dat $DAT_FILE
EOF
chmod +x myhpl/run-baremetal.sh

# ---- Launch ----------------------------------------------------------------
srun --mpi=pmix --export=ALL -n $SLURM_NTASKS \
    ./myhpl/run-baremetal.sh

echo "---------------- HPL Bare-metal Run End $(date) ------------------"
echo "==============================================================================="
