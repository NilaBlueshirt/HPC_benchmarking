#!/bin/bash
# build_hpl_nvblas.sh - LAST-RESORT HPL for a mixed GPU fleet: netlib HPL 2.3 with NVIDIA's
# NVBLAS intercepting DGEMM. Runs on any CUDA GPU, A100 and H100 side by side, with no
# NVSHMEM, NCCL or UCX tuning involved. It is also 3 to 5 times slower per GPU than the
# NVIDIA container, because only DGEMM is offloaded and every call crosses PCIe, so it
# cannot reach a TOP500-listable number on Sol. Use it to prove the fabric and the mixed
# job layout work, or as a diagnostic, not for the submitted run.
#
# Prerequisites on the build host: an MPI compiler wrapper (mpicc), a CPU BLAS (OpenBLAS),
# and a CUDA toolkit that ships libnvblas.so. Set the three variables below to your module
# names or paths, then:  bash alt/build_hpl_nvblas.sh
set -euo pipefail

MPI_MODULE=${MPI_MODULE:-openmpi}          # module providing mpicc
BLAS_MODULE=${BLAS_MODULE:-openblas}       # module providing libopenblas.so
CUDA_MODULE=${CUDA_MODULE:-cuda}           # module providing $CUDA_HOME/lib64/libnvblas.so
PREFIX=${PREFIX:-$HOME/hpl-nvblas}

module load "$MPI_MODULE" "$BLAS_MODULE" "$CUDA_MODULE" 2>/dev/null || echo "module load skipped; relying on current environment"
command -v mpicc >/dev/null || { echo "mpicc not found" >&2; exit 1; }
OPENBLAS_LIB=${OPENBLAS_LIB:-$(ldconfig -p 2>/dev/null | awk '/libopenblas.so / {print $NF; exit}')}
[ -n "${OPENBLAS_LIB:-}" ] || OPENBLAS_LIB=$(find "${OPENBLAS_ROOT:-${OPENBLAS_HOME:-/usr}}" -name 'libopenblas.so' 2>/dev/null | head -1)
[ -n "${OPENBLAS_LIB:-}" ] || { echo "set OPENBLAS_LIB=/path/to/libopenblas.so" >&2; exit 1; }
CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}
[ -e "$CUDA_HOME/lib64/libnvblas.so" ] || { echo "libnvblas.so not under $CUDA_HOME/lib64" >&2; exit 1; }

mkdir -p "$PREFIX" && cd "$PREFIX"
[ -e hpl-2.3.tar.gz ] || curl -fsSLO https://netlib.org/benchmark/hpl/hpl-2.3.tar.gz
[ -d hpl-2.3 ] || tar xzf hpl-2.3.tar.gz
cd hpl-2.3

cat > Make.nvblas <<MAKE
SHELL        = /bin/sh
CD           = cd
CP           = cp
LN_S         = ln -s
MKDIR        = mkdir
RM           = /bin/rm -f
TOUCH        = touch
ARCH         = nvblas
TOPdir       = $PREFIX/hpl-2.3
INCdir       = \$(TOPdir)/include
BINdir       = \$(TOPdir)/bin/\$(ARCH)
LIBdir       = \$(TOPdir)/lib/\$(ARCH)
HPLlib       = \$(LIBdir)/libhpl.a
MPdir        =
MPinc        =
MPlib        =
LAdir        = $(dirname "$OPENBLAS_LIB")
LAinc        =
LAlib        = $OPENBLAS_LIB
F2CDEFS      =
HPL_INCLUDES = -I\$(INCdir) -I\$(INCdir)/\$(ARCH) \$(LAinc) \$(MPinc)
HPL_LIBS     = \$(HPLlib) \$(LAlib) \$(MPlib)
HPL_OPTS     = -DHPL_CALL_CBLAS -DHPL_PROGRESS_REPORT
HPL_DEFS     = \$(F2CDEFS) \$(HPL_OPTS) \$(HPL_INCLUDES)
CC           = mpicc
CCNOOPT      = \$(HPL_DEFS)
CCFLAGS      = \$(HPL_DEFS) -O3 -march=native -fopenmp
LINKER       = mpicc
LINKFLAGS    = \$(CCFLAGS)
ARCHIVER     = ar
ARFLAGS      = r
RANLIB       = echo
MAKE

make arch=nvblas -j8
ls -l bin/nvblas/xhpl

# NVBLAS configuration: one GPU per rank is selected by CUDA_VISIBLE_DEVICES, so ALL is right.
cat > "$PREFIX/nvblas.conf" <<CONF
NVBLAS_LOGFILE       /dev/null
NVBLAS_CPU_BLAS_LIB  $OPENBLAS_LIB
NVBLAS_GPU_LIST      ALL
NVBLAS_TILE_DIM      4096
NVBLAS_AUTOPIN_MEM_ENABLED
CONF

cat <<DONE

Built $PREFIX/hpl-2.3/bin/nvblas/xhpl

Run it from a Slurm step with one rank per GPU (the hetjob headers in the repo work unchanged;
replace the apptainer line in mix_common.sh with the srun below, and point HPL.dat at the dat file):

  export NVBLAS_CONFIG_FILE=$PREFIX/nvblas.conf
  export LD_PRELOAD=$CUDA_HOME/lib64/libnvblas.so
  export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
  cp dats/256GPU_grid.dat HPL.dat
  srun --het-group=0-1 --mpi=pmix $PREFIX/hpl-2.3/bin/nvblas/xhpl

Expect on the order of 3 to 6 TFlop/s per GPU, not the 11.7 of the NVIDIA container.
DONE
