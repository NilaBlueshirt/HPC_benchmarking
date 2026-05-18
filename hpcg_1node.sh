#!/bin/bash
#SBATCH -J hpcg_1node
#SBATCH -p hpl
#SBATCH -q public
#SBATCH -t 0-1
#SBATCH -C a100_80
#SBATCH -N 1-1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=single:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH -o hpcg_1node.log
#SBATCH --open-mode=append
#SBATCH --distribution=block:block
#SBATCH --export=NONE

echo "==============================================================================="
echo "---------------- HPCG 1-node Run Start $(date) ------------------"
echo ""

DAT_FILE=$(readlink -f ./dats/HPCG_1node.dat)
DAT_MOUNT="$(dirname "$DAT_FILE"):/my-dat-files"
SIF=/packages/apps/simg/nvhpl_25.09.sif

echo "SBATCH script for JOB ${SLURM_JOB_ID} :"
scontrol write batch_script $SLURM_JOB_ID - | grep -v '^ *echo '
echo ""
echo "Using DAT File: ${DAT_FILE}"
cat ${DAT_FILE}
echo "Starting hpcg container"

mkdir -p myhpcg
cat > myhpcg/run.sh <<EOF
#!/bin/bash
GPU_N=\${SLURM_GPUS_ON_NODE:-\$(nvidia-smi -L 2>/dev/null | wc -l || echo 4)}

echo "SLURM_NNODES=$SLURM_NNODES SLURM_NTASKS=$SLURM_NTASKS NODELIST=$SLURM_JOB_NODELIST"
echo "On node \$(hostname): SLURM_GPUS_ON_NODE=\$SLURM_GPUS_ON_NODE CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES"

export OMP_NUM_THREADS=12
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export MONITOR_GPU=1
export GPU_TEMP_WARNING=80
export GPU_CLOCK_WARNING=562
/workspace/hpcg.sh --dat "/my-dat-files/$(basename "$DAT_FILE")"
EOF
SCRIPT_FILE=$(readlink -f ./myhpcg/run.sh)
SCRIPT_MOUNT="$(dirname "$SCRIPT_FILE"):/myhpcg"
chmod +x $SCRIPT_FILE


srun --mpi=pmix --export=ALL  -n $SLURM_NTASKS \
    apptainer run --nv -B "${DAT_MOUNT}" -B "${SCRIPT_MOUNT}" "$SIF" /myhpcg/run.sh



echo "---------------- HPCG 1-node Run End $(date) ------------------"
echo "==============================================================================="
