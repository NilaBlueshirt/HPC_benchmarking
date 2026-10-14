#!/bin/bash
#SBATCH -J hplmxp_240
#SBATCH --reservation=oct2026_maint
#SBATCH -p hpl
#SBATCH -q public
#SBATCH -t 0-8
#SBATCH -N 60-60
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=single:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=512000M
#SBATCH -o hplmxp.log
#SBATCH --open-mode=append
#SBATCH --distribution=block:block
#SBATCH --export=NONE

# ---- HPL-MxP parameters ------------------------------------------------------
# hpl-mxp.sh takes command-line flags; it has no dat file (NVIDIA HPC-Benchmarks docs).
# N     1484800 = 725 x 2048.  FP16 working copy on GPU = N^2 x 2 B / 240 = 18.4 GB/GPU.
#       FP64 matrix = N^2 x 8 B = 17.6 TB total.  --fill-device 1 places as much of it
#       as fits on each GPU (leaving 3 GB free); the remainder, ~75 GB per node, sits in
#       host RAM.  Without --fill-device the whole FP64 matrix would need ~295 GB/node.
# NB    2048, NVIDIA's HPL-MxP example value.
# Grid  15 x 16 column-major: the grid and PMAP=1 ordering of the 2.815 PFlop/s HPL run (16 x 15 lost ~10%).
N=1484800
NB=2048
NPROW=15
NPCOL=16
NPORDER=column
SIF=/packages/apps/simg/nvhpl_26.02.01.sif
# -------------------------------------------------------------------------------

echo "==============================================================================="
echo "---------------- HPL-MxP Run Start $(date) ------------------"
echo ""
echo "SBATCH script for JOB ${SLURM_JOB_ID} :"
scontrol write batch_script $SLURM_JOB_ID - | grep -v '^ *echo '
echo ""
echo "HPL-MxP params: N=$N NB=$NB grid=${NPROW}x${NPCOL} ${NPORDER}-major  SIF=$SIF"
echo "Starting hpl-mxp container"

mkdir -p myhplmxp
cat > myhplmxp/run.sh <<EOS
#!/bin/bash
HCA=\$(basename /sys/class/net/ib0/device/infiniband/* 2>/dev/null || ibdev2netdev | awk "/ ib0 /{print \$1; exit}")
MAP=\${HCA}:1

echo "SLURM_NNODES=$SLURM_NNODES SLURM_NTASKS=$SLURM_NTASKS NODELIST=$SLURM_JOB_NODELIST"
echo "On node \$(hostname): SLURM_GPUS_ON_NODE=\$SLURM_GPUS_ON_NODE CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES"

export OMP_NUM_THREADS=12
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMPI_MCA_pml=ucx
export OMPI_MCA_coll=^ucc
export OMPI_MCA_btl=^openib
export PMIX_MCA_gds=hash
export UCX_TLS=dc_x,sm,self,cuda_ipc,cuda_copy,gdr_copy
export UCX_NET_DEVICES=\$MAP
export UCX_SOCKADDR_TLS_PRIORITY=rdmacm
export UCX_POSIX_USE_PROC_LINK=n
export UCX_IB_GDR_DIRECT_RDMA=y
export NCCL_DEBUG=WARN
export NCCL_NET=IB
export NCCL_COLLNET_ENABLE=1
export NCCL_IB_HCA=\$HCA
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_GID_INDEX=0
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_CROSS_NIC=0
# NCCL_P2P_DISABLE is intentionally not set: HPL-MxP's panel broadcast runs on NCCL,
# and the 16-GPU HPL test gained 7% when it was removed from the NCCL path.

# hpl-mxp.sh requires --gpu-affinity ("List of GPUs is not defined" without it) and applies
# it by exporting CUDA_VISIBLE_DEVICES=<list[local rank]>. A fixed 0:0:0:0 overwrote the
# per-rank value Slurm sets (3,1,2,0) and put all 4 ranks on GPU 0 (hung 3+ h, 2026-10-14).
# Repeating this rank's own Slurm value makes that export a no-op whatever index it picks.
GPU=\${CUDA_VISIBLE_DEVICES:-0}
/workspace/hpl-mxp.sh --n $N --nb $NB --nprow $NPROW --npcol $NPCOL --nporder $NPORDER --gpu-affinity \$GPU:\$GPU:\$GPU:\$GPU --fill-device 1 --monitor-gpu 1 --monitor-gpu-temp-warning 80 --monitor-gpu-clock-warning 562
EOS
SCRIPT_FILE=$(readlink -f ./myhplmxp/run.sh)
SCRIPT_MOUNT="$(dirname "$SCRIPT_FILE"):/myhplmxp"
chmod +x $SCRIPT_FILE


srun --mpi=pmix --export=ALL  -n $SLURM_NTASKS \
    apptainer run --nv -B "${SCRIPT_MOUNT}" "$SIF" /myhplmxp/run.sh



echo "---------------- HPL-MxP Run End $(date) ------------------"
echo "==============================================================================="
