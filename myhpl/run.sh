#!/bin/bash
HCA=$(basename /sys/class/net/ib0/device/infiniband/* 2>/dev/null || ibdev2netdev | awk "/ ib0 /{print $1; exit}")
MAP=${HCA}:1
GPU_N=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 4)}
PE_MAP=$(printf "%s," $(yes "$MAP" | head -n "$GPU_N")); PE_MAP=${PE_MAP%,}

echo "SLURM_NNODES=2 SLURM_NTASKS=8 NODELIST=sg[001-002]"
echo "On node $(hostname): SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES PE_MAP=$PE_MAP"

#export HPL_OOC_MODE=1

export OMP_NUM_THREADS=12
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMPI_MCA_pml=ucx
export OMPI_MCA_coll=^ucc
export OMPI_MCA_btl=^openib
export PMIX_MCA_gds=hash
export UCX_TLS=dc_x,sm,self,cuda_ipc,cuda_copy,gdr_copy
export UCX_NET_DEVICES=$MAP
export UCX_SOCKADDR_TLS_PRIORITY=rdmacm
export UCX_POSIX_USE_PROC_LINK=n
export UCX_IB_GDR_DIRECT_RDMA=y
export HPL_USE_NVSHMEM=1
export HPL_NVSHMEM_SWAP=1
export HPL_ALLOC_HUGEPAGES=1
export NVSHMEM_REMOTE_TRANSPORT=ibrc 
export NVSHMEM_HCA_PE_MAPPING=$PE_MAP
export NVSHMEM_FORCE_IBGDA=1
export NVSHMEM_DEBUG=WARN
export NCCL_DEBUG=WARN
export NCCL_NET=IB
export NCCL_COLLNET_ENABLE=1
export NCCL_IB_HCA=$HCA
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_GID_INDEX=0
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_CROSS_NIC=0
export NCCL_P2P_DISABLE=1
export MONITOR_GPU=1
export GPU_TEMP_WARNING=80
export GPU_CLOCK_WARNING=562
/workspace/hpl.sh --dat "/my-dat-files/2nodes.dat"
