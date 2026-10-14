#!/bin/bash
#SBATCH -J HPL-MxP-bm-224
#SBATCH --reservation=maint
#SBATCH -p hpl
#SBATCH -q public
#SBATCH -t 0-1
#SBATCH -C a100_80
#SBATCH -N 56-56
#SBATCH -w sg[001-050],sg[235-239],scg0[09-12]
#SBATCH -x sg034,scg009,sdg051
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH -o hpl-mxp-baremetal.log
#SBATCH --open-mode=append
#SBATCH --distribution=block:block

echo "==============================================================================="
echo "---------------- HPL-MxP Bare-metal Run Start $(date) ------------------"
echo ""

# ---- Load modules ----------------------------------------------------------
module purge
module load nvhpc/26.3-cuda-13.1-nompi
module load hpcx/2.25.1

# ---- HPL-MxP package location ----------------------------------------------
# Same parent v26.02 extraction as HPL; just a different sub-dir.
# Expected layout:
#   cuda13/hpl-mxp-linux-x86_64/hpl-mxp.sh
#   cuda13/hpl-mxp-linux-x86_64/xhpl-mxp
#   cuda13/hpl-mxp-linux-x86_64/sample-dat/
HPLDIR=${HPLDIR:-$HOME/HPL/nv_binary/nvidia_hpc_benchmarks_openmpi-linux-x86_64-26.02.02-archive/cuda13}
MXPBIN=${MXPBIN:-${HPLDIR}/hpl-mxp-linux-x86_64}

echo "=== Loaded modules ==="
module list 2>&1
echo ""
echo "=== HPL-MxP package check ==="
echo "MXPBIN=${MXPBIN}"
ls -la "${MXPBIN}/hpl-mxp.sh" 2>&1 || { echo "ERROR: hpl-mxp.sh not at ${MXPBIN}/hpl-mxp.sh — adjust MXPBIN. Try: ls ${HPLDIR}"; exit 1; }
ls -la "${MXPBIN}/xhpl_mxp"   2>&1 || { echo "ERROR: xhpl_mxp binary missing under ${MXPBIN}"; exit 1; }
echo ""

DAT_FILE=${DAT_FILE:-${MXPBIN}/sample-dat/HPL-MxP-224GPU.dat}

echo "SBATCH script for JOB ${SLURM_JOB_ID} :"
scontrol write batch_script $SLURM_JOB_ID - | grep -v '^ *echo '
echo ""
echo "Using DAT File: ${DAT_FILE}"
cat ${DAT_FILE} 2>&1 || { echo "ERROR: DAT file missing at ${DAT_FILE}"; exit 1; }
echo ""

# ---- Per-rank wrapper ------------------------------------------------------
mkdir -p myhpl
cat > myhpl/run-mxp-baremetal.sh <<EOF
#!/bin/bash
# Re-load modules in this child shell.
source /etc/profile.d/lmod.sh 2>/dev/null || source /etc/profile.d/modules.sh 2>/dev/null
module purge >/dev/null 2>&1
module load nvhpc/26.3-cuda-13.1-nompi
module load hpcx/2.25.1
export LD_LIBRARY_PATH=/packages/apps/hpcx/2.25.1/inbox/ucx/lib:$LD_LIBRARY_PATH

# CRITICAL: the Sol hpcx module sets up OpenMPI/HCOLL/UCC paths but DOES NOT add
# UCX's lib dir to LD_LIBRARY_PATH. That makes mca_pml_ucx.so fail to load.
# Two-pronged fix: try hpcx-init.sh (canonical), and also search for libucp.so
# under /packages/apps/hpcx and add its parent dir as a fallback.
HPCX_INIT=/packages/apps/hpcx/2.25.1/inbox/hpcx-init.sh
[ -f "\$HPCX_INIT" ] && source "\$HPCX_INIT" 2>/dev/null && (hpcx_load 2>/dev/null || true)
UCX_LIB=\$(find /packages/apps/hpcx/2.25.1 -name "libucp.so*" 2>/dev/null | head -1 | xargs -r dirname)
[ -n "\$UCX_LIB" ] && export LD_LIBRARY_PATH=\$UCX_LIB:\$LD_LIBRARY_PATH

# Diagnostic — first rank dumps env so we can confirm UCX is now resolved
if [ "\$SLURM_PROCID" = "0" ]; then
  echo "DEBUG (rank 0): which mpirun = \$(which mpirun 2>&1)"
  echo "DEBUG (rank 0): UCX_LIB found at = \$UCX_LIB"
  echo "DEBUG (rank 0): libucp.so resolution test:"
  ldconfig -p 2>/dev/null | grep libucp || echo "  not in ldconfig cache (using LD_LIBRARY_PATH)"
  ls \$UCX_LIB/libucp.so* 2>&1 | head -3
  echo "DEBUG (rank 0): pml plugins available:"
  ompi_info --param pml all 2>&1 | grep -E "MCA pml" | head -5
fi

HCA=\$(basename /sys/class/net/ib0/device/infiniband/* 2>/dev/null || ibdev2netdev | awk "/ ib0 /{print \\\$1; exit}")
MAP=\${HCA}:1
GPU_N=\${SLURM_GPUS_ON_NODE:-\$(nvidia-smi -L 2>/dev/null | wc -l || echo 4)}
PE_MAP=\$(printf "%s," \$(yes "\$MAP" | head -n "\$GPU_N")); PE_MAP=\${PE_MAP%,}
# UCX affinity: same HCA repeated per local GPU (matches NVSHMEM_HCA_PE_MAPPING)
UCX_AFF=\$(printf "%s:" \$(yes "\$HCA" | head -n "\$GPU_N")); UCX_AFF=\${UCX_AFF%:}

echo "Host=\$(hostname) SLURM_PROCID=\$SLURM_PROCID SLURM_LOCALID=\$SLURM_LOCALID CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES PE_MAP=\$PE_MAP UCX_AFF=\$UCX_AFF"

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

# NVSHMEM — ON (bare-metal dodges the container's NVSHMEM bootstrap bug)
# HPL-MxP-specific env vars:
export HPL_MXP_USE_NVSHMEM=1
export HPL_MXP_NVSHMEM_SWAP=1
export HPL_MXP_NVSHMEM_INIT=0
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
# Note: NCCL_P2P_DISABLE intentionally NOT set — we want intra-node NVLink P2P.

# Monitoring
export MONITOR_GPU=1
export GPU_TEMP_WARNING=80
export GPU_CLOCK_WARNING=562

# Run HPL-MxP
# HPL-MxP launcher in v26.02 doesn't read .dat fully — pass everything as flags.
# Matches NVIDIA's reference invocation pattern from sample-slurm.
# Affinity flags: only the uniform ones (gpu, ucx). Skipping cpu/mem-affinity
# because NUMA topology differs across some nodes in the allocation.
${MXPBIN}/hpl-mxp.sh \
    --gpu-affinity 0:1:2:3 \
    --ucx-affinity \${UCX_AFF} \
    --n 1605632 --nb 2048 --nprow 14 --npcol 16 --nporder row \
    --preset-gemm-kernel 0 \
    --u-panel-chunk-nbs 16 \
    --use-mpi-panel-broadcast 50 \
    --sloppy-type 2 \
    --call-dgemv-with-multiple-threads 0 \
    --Anq-device 0 \
    --mpi-use-mpi 1 \
    --prioritize-trsm 0 \
    --prioritize-factorization 1
EOF
chmod +x myhpl/run-mxp-baremetal.sh

# ---- Launch ----------------------------------------------------------------
# Back to srun --mpi=pmix. mpirun lost the unlimited memlock that
# Slurm-launched processes inherit from slurmstepd, causing EPERM on
# `ibv_modify_qp → INIT` during UCX bootstrap.
# The module-reload inside run-mxp-baremetal.sh should solve the previous
# "PML ucx cannot be selected" srun-side issue.
srun --mpi=pmix --export=ALL -n $SLURM_NTASKS \
    ./myhpl/run-mxp-baremetal.sh

echo "---------------- HPL-MxP Bare-metal Run End $(date) ------------------"
echo "==============================================================================="
