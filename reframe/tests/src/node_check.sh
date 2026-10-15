#!/bin/bash
# node_check.sh - pre-release check of one Sol node, run as the body of a whole-node Slurm job:
# identity, accelerator count, srun launch, MPI start-up, STREAM bandwidth and, on NVIDIA nodes,
# PCIe/HBM/GEMM on every GPU. Prints SOL_* result lines and always ends with SOL_DONE; a step
# that fails or hangs never stops the others. SolNodeCheck (tests/node_checks.py) submits it
# through ReFrame and checks the output.
#
# By hand: copy this directory to a filesystem the node can see (the script builds its binaries
# in the current directory), cd there and submit from a clean shell (module purge), for example:
#   SOL_NODE=scg017 SOL_CORES=48 SOL_ACCEL=nvidia SOL_GPU_COUNT=4 SOL_MOD_GCC=gcc/16.1.0 \
#   SOL_MOD_CUDA=cuda-13.2.1-gcc-13.2.0 SOL_MOD_NODE_MPI=openmpi/5.0.8-gcc-15.2.0 \
#   sbatch -w scg017 --exclusive --mem=0 -n 48 --ntasks-per-node=48 --gres=gpu:l40:4 \
#          -p general -q private --reservation=oct2026_maint -t 20 -o node_check_%j.out node_check.sh
#
# SOL_ACCEL is nvidia, amd, gaudi or none. An empty SOL_MOD_NODE_MPI or SOL_MOD_CUDA skips that
# step (SOL_SKIP line); an empty SOL_MOD_GCC builds STREAM with the system gcc.

SOL_NODE=${SOL_NODE:-$(hostname -s)}
SOL_CORES=${SOL_CORES:-${SLURM_NTASKS:-1}}
SOL_ACCEL=${SOL_ACCEL:-none}
SOL_GPU_COUNT=${SOL_GPU_COUNT:-0}
SOL_SRUN_MPI=${SOL_SRUN_MPI:-pmix}
SOL_STREAM_ELEMS=${SOL_STREAM_ELEMS:-268435456}
SOL_GPU_SECONDS=${SOL_GPU_SECONDS:-20}
ARCH=$(uname -m)
exec 3>&1    # the job's stdout, for SOL_TIMEOUT lines from commands whose output is redirected

# run_to STEP SECONDS CMD...: run CMD with a time limit and print SOL_TIMEOUT if it is hit
run_to() {
    local step=$1 limit=$2 t0=$SECONDS rc
    shift 2
    timeout -k 30 "$limit" "$@"
    rc=$?
    if [ "$rc" -eq 124 ] || { [ "$rc" -eq 137 ] && [ $((SECONDS - t0)) -ge "$limit" ]; }; then
        echo "SOL_TIMEOUT step=$step seconds=$limit" >&3
    fi
    return "$rc"
}

# load_module NAME: module load into the current (sub)shell, setting up Lmod first if needed
load_module() {
    # shellcheck source=/dev/null
    type module >/dev/null 2>&1 || source /etc/profile.d/lmod.sh >/dev/null 2>&1 ||
        source /usr/share/lmod/lmod/init/bash >/dev/null 2>&1
    if ! type module >/dev/null 2>&1; then
        echo "node_check.sh: no module command, cannot load $1" >&2
        return 1
    fi
    if ! module load "$1" >&2; then
        echo "node_check.sh: module load $1 failed" >&2
        return 1
    fi
}

# cuda_host_gcc: nvcc 13.2 rejects gcc > 15. Unload a newer gcc module inherited from the
# submitting shell; if gcc on PATH is still too new, point nvcc at the system g++ (sets ccbin).
cuda_host_gcc() {
    local v
    v=$(gcc -dumpversion 2>/dev/null); v=${v%%.*}
    case $v in ''|*[!0-9]*) return 0 ;; esac
    [ "$v" -le 15 ] && return 0
    echo "node_check.sh: gcc $v on PATH is too new for nvcc, unloading it" >&2
    module unload gcc >&2
    v=$(gcc -dumpversion 2>/dev/null); v=${v%%.*}
    if ! [ "$v" -le 15 ] 2>/dev/null; then
        echo "node_check.sh: gcc on PATH is ${v:-gone} after the unload, nvcc uses /usr/bin/g++" >&2
        ccbin=/usr/bin/g++
    fi
}

# modkey KEY: the site.conf key that sets this module for this node's architecture
modkey() {
    if [ "$ARCH" = aarch64 ]; then echo "${1}_AARCH64"; else echo "$1"; fi
}

step_ident() {
    local slurmd mem drv
    slurmd=$({ slurmd -V || /usr/sbin/slurmd -V; } 2>/dev/null | awk 'NR == 1 {print $2}')
    mem=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null)
    echo "SOL_IDENT host=$(hostname -s) arch=$ARCH nproc=$(nproc) mem_kb=${mem:-NA}" \
         "kernel=$(uname -r) slurmd=${slurmd:-NA}"
    if [ "$SOL_ACCEL" = nvidia ]; then
        # through a file: an nvidia-smi stuck in the driver survives the kill and holds any pipe
        run_to ident 60 nvidia-smi --query-gpu=driver_version --format=csv,noheader > smi_driver.txt 2>/dev/null
        drv=$(head -1 smi_driver.txt 2>/dev/null | tr -d ' ')
        case $drv in ''|*[!0-9.]*) drv=NA ;; esac
        echo "SOL_DRIVER nvidia=$drv"
    fi
}

step_devices() {
    local n
    n=$(run_to devices 120 bash accel_count.sh "$SOL_ACCEL")
    case $n in ''|*[!0-9]*) n=0 ;; esac
    echo "SOL_DEVICES kind=$SOL_ACCEL count=$n expected=$SOL_GPU_COUNT"
    if [ "$n" != "$SOL_GPU_COUNT" ] && [ "$SOL_ACCEL" = nvidia ]; then
        timeout -k 10 60 nvidia-smi -L >&2
    fi
}

step_srun() {
    local rc tasks hosts
    run_to srun 180 srun -n "$SOL_CORES" --ntasks-per-node="$SOL_CORES" hostname -s > srun_hosts.txt
    rc=$?
    tasks=$(grep -c . srun_hosts.txt)
    hosts=$(sort -u srun_hosts.txt | grep -c .)
    echo "SOL_SRUN tasks=$tasks hosts=$hosts expected=$SOL_CORES rc=$rc"
}

step_mpi_hello() {
    local ok=0
    if [ -z "$SOL_MOD_NODE_MPI" ]; then
        echo "SOL_SKIP step=mpi_hello reason=\"no MPI module configured for $ARCH" \
             "(set $(modkey MOD_NODE_MPI) in site.conf)\""
        return
    fi
    rm -f mpi_hello
    if load_module "$SOL_MOD_NODE_MPI" && run_to mpi_hello 300 mpicc -O2 -o mpi_hello mpi_hello.c >&2; then
        ok=1
    fi
    echo "SOL_BUILD step=mpi_hello ok=$ok module=\"$SOL_MOD_NODE_MPI\""
    if [ "$ok" = 1 ]; then
        run_to mpi_hello 300 srun --mpi="$SOL_SRUN_MPI" -n "$SOL_CORES" --ntasks-per-node="$SOL_CORES" ./mpi_hello
    fi
}

step_stream() {
    local ok=0 label=${SOL_MOD_GCC:-system} march=-march=native
    case $ARCH in aarch64|arm64) march=-mcpu=native ;; esac
    rm -f stream_omp
    if { [ -z "$SOL_MOD_GCC" ] || load_module "$SOL_MOD_GCC"; } &&
       run_to stream 300 gcc -O3 -fopenmp "$march" -o stream_omp stream_omp.c >&2; then
        ok=1
    fi
    echo "SOL_BUILD step=stream ok=$ok module=\"$label\""
    if [ "$ok" = 1 ]; then
        export OMP_NUM_THREADS=$SOL_CORES OMP_PLACES=cores OMP_PROC_BIND=spread
        run_to stream 600 ./stream_omp "$SOL_STREAM_ELEMS" 10
    fi
}

# CUDA step: load only the CUDA module (never MOD_GCC: nvcc rejects a too new host gcc)
step_gpu() {
    local ok=0 cudalib ccbin=
    if [ "$SOL_ACCEL" != nvidia ]; then
        return
    fi
    if [ -z "$SOL_MOD_CUDA" ]; then
        echo "SOL_SKIP step=gpu reason=\"no CUDA module configured for $ARCH" \
             "(set $(modkey MOD_CUDA) in site.conf)\""
        return
    fi
    rm -f gpu_check
    if load_module "$SOL_MOD_CUDA" && cuda_host_gcc &&
       run_to gpu 300 nvcc ${ccbin:+-ccbin "$ccbin"} -O3 -arch=native -o gpu_check gpu_check.cu -lcublas >&2; then
        ok=1
    fi
    echo "SOL_BUILD step=gpu ok=$ok module=\"$SOL_MOD_CUDA\""
    if [ "$ok" = 1 ]; then
        # gpu_check loads libcublas at run time; Spack-made modules often leave LD_LIBRARY_PATH alone
        cudalib=$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")/lib64
        if [ -d "$cudalib" ]; then
            export LD_LIBRARY_PATH=$cudalib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
        fi
        bash run_gpu.sh "$SOL_GPU_COUNT" "$SOL_GPU_SECONDS"
    fi
}

echo "node_check.sh: node $SOL_NODE, $SOL_CORES cores, accel $SOL_ACCEL, in $PWD" >&2
for step in ident devices srun mpi_hello stream gpu; do
    echo "== $step $(date '+%F %T')" >&2
    ( "step_$step" )
done
echo "SOL_DONE"
