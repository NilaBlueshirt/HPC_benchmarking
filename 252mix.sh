#!/bin/bash
#SBATCH --reservation=maint
#SBATCH -p hpl
#SBATCH -q public
#SBATCH -t 0-4
#SBATCH -N 63-63
#SBATCH -w sdg051,scg[005,009-012],sg[001-050,235-239],scg006,scg020,scg024
#SBATCH -C a100_80|h100
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=single:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH -o mix.log
#SBATCH --open-mode=append
#SBATCH --distribution=block:block
#SBATCH --export=NONE

echo "==============================================================================="
echo "---------------- HPL Run Start $(date) ------------------"
echo ""

DAT_FILE=$(readlink -f ./dats/252GPU_grid.dat)
#DAT_FILE=$(readlink -f ./dats/8GPU.dat)
DAT_MOUNT="$(dirname "$DAT_FILE"):/my-dat-files"
SIF=/packages/apps/simg/nvhpl_25.09.sif

echo "SBATCH script for JOB ${SLURM_JOB_ID} :"
scontrol write batch_script $SLURM_JOB_ID - | grep -v '^ *echo '
echo ""
echo "Using DAT File: ${DAT_FILE}"
cat ${DAT_FILE}
echo "Starting hpl container"

mkdir -p myhpl
cat > myhpl/run.sh <<EOF
#!/bin/bash
HCA=\$(basename /sys/class/net/ib0/device/infiniband/* 2>/dev/null || ibdev2netdev | awk "/ ib0 /{print \$1; exit}")
MAP=\${HCA}:1
GPU_N=\${SLURM_GPUS_ON_NODE:-\$(nvidia-smi -L 2>/dev/null | wc -l || echo 4)}
PE_MAP=\$(printf "%s," \$(yes "\$MAP" | head -n "\$GPU_N")); PE_MAP=\${PE_MAP%,}

echo "SLURM_NNODES=$SLURM_NNODES SLURM_NTASKS=$SLURM_NTASKS NODELIST=$SLURM_JOB_NODELIST"
echo "On node \$(hostname): SLURM_GPUS_ON_NODE=\$SLURM_GPUS_ON_NODE CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES PE_MAP=\$PE_MAP"

export OMP_NUM_THREADS=12
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMPI_MCA_pml=ucx
export OMPI_MCA_coll=^ucc
export OMPI_MCA_btl=^openib
export PMIX_MCA_gds=hash
#export UCX_TLS=dc_x,sm,self,cuda_ipc,cuda_copy,gdr_copy
export UCX_TLS=rc_x,sm,self,cuda_copy,gdr_copy
export UCX_NET_DEVICES=\$MAP
export UCX_SOCKADDR_TLS_PRIORITY=rdmacm
export UCX_POSIX_USE_PROC_LINK=0
#export UCX_IB_GDR_DIRECT_RDMA=0
export HPL_USE_NVSHMEM=0
export HPL_NVSHMEM_SWAP=0
#unset NVSHMEM_REMOTE_TRANSPORT
#unset NVSHMEM_HCA_PE_MAPPING
#unset NVSHMEM_FORCE_IBGDA
#unset NVSHMEM_DEBUG
export NVSHMEM_REMOTE_TRANSPORT=ibrc 
export NVSHMEM_HCA_PE_MAPPING=\$PE_MAP
#export NVSHMEM_FORCE_IBGDA=0
#export NVSHMEM_DEBUG=WARN
#export HPL_ALLOC_HUGEPAGES=0
export NCCL_DEBUG=WARN
export NCCL_NET=IB
#export NCCL_COLLNET_ENABLE=1
export NCCL_IB_HCA=\$HCA
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_GID_INDEX=0
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_CROSS_NIC=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_NCHANNELS=8
export NCCL_NET_GDR_LEVEL=PHB
export MONITOR_GPU=1
export GPU_TEMP_WARNING=80
export GPU_CLOCK_WARNING=562
/workspace/hpl.sh --dat "/my-dat-files/$(basename "$DAT_FILE")"
EOF
SCRIPT_FILE=$(readlink -f ./myhpl/run.sh)
SCRIPT_MOUNT="$(dirname "$SCRIPT_FILE"):/myhpl"
chmod +x $SCRIPT_FILE


srun --mpi=pmix --export=ALL  -n $SLURM_NTASKS \
    apptainer run --nv -B "${DAT_MOUNT}" -B "${SCRIPT_MOUNT}" "$SIF" /myhpl/run.sh



echo "---------------- HPL Run End $(date) ------------------"
echo "==============================================================================="