# Mixed A100 + H100 HPL on Sol

Run files and procedure for adding the H100 nodes to the 240-A100 HPL run. Written for the
Sol admin running the October 2026 window. Everything here runs on the cluster; nothing was
executed there while writing it, so treat the first smoke test as the test of this guide too.

## 1. Is there an alternative to the NVIDIA container for H100?

Not a faster one. The NVIDIA HPC Benchmarks container (`nvhpl_25.09.sif`) is the HPL every
H100 entry on the TOP500 uses, and it already works on your H100s: the single-node test in
`hpl-score.csv` reached 45 TFlop/s per H100 with v25.09. What failed in May was the mixed
252-GPU job, and that job ran with NVSHMEM off, `rc_x` instead of `dc_x` and no CUDA IPC,
while the 8-GPU mixed pairs with the normal settings ran at full A100 speed. The problem to
solve is the mixed job layout, not H100 support.

Options, in the order worth trying:

| Option | What changes | Cost in score |
|---|---|---|
| Same container, hetjob layout (`256mix.sh`, `260mix.sh`) | per-GPU HCA mapping on 8-way nodes, 240-run transport settings kept | none; this is the target |
| Same container, NVSHMEM off (`MIX_NVSHMEM=0`) | HPL falls back to its NCCL/MPI path | a few percent on A100 in your May tests |
| Other container release (`hpc-benchmarks:24.09`, `25.04`) | different NVSHMEM/UCX builds | none if it runs; pull with `apptainer pull docker://nvcr.io/nvidia/hpc-benchmarks:<tag>` |
| netlib HPL 2.3 + NVBLAS (`alt/build_hpl_nvblas.sh`) | no NVSHMEM, NCCL or UCX at all; only DGEMM on the GPU | 3 to 5 times slower per GPU; cannot make the list; diagnostic only |

## 2. Files

| File | Purpose |
|---|---|
| `256mix.sh` | 240 A100 + 16 H100 (two 8-way nodes), 16x16 grid |
| `260mix.sh` | 240 A100 + 20 H100 (two 8-way + one 4-way), 13x20 grid |
| `mix_test12.sh` | 1 A100 node + 1 8-way node, 12 GPUs, 3x4, about 2 minutes of compute |
| `mix_test32.sh` | 4 A100 nodes + 2 8-way nodes, 32 GPUs, 4x8, about 4 minutes |
| `mix_common.sh` | shared body: builds the per-rank `run.sh`, launches `srun --het-group` |
| `h100_mix_check.sh` | pre-flight for the H100 nodes (section 4) |
| `dats/256GPU_mix.dat`, `dats/260GPU_mix.dat` | HPL inputs, 74 GB per GPU (86% fill; the 240 dat's first N fills 92.7%), NB 448, column-major, and the single algorithm line of `HPL_240GPU.dat` (PFACT 0, NBMIN 2, NDIV 2, RFACT 0, BCAST 3, DEPTH 1). The 72-run sweep they shipped with lives on as `dats/256GPU_mix_sweep.dat` (`DAT=dats/256GPU_mix_sweep.dat sbatch 256mix.sh`) |
| `dats/12GPU_mix_test.dat`, `dats/32GPU_mix_test.dat` | inputs for the two test scripts |
| `dats/HPL-MxP_256GPU.dat`, `dats/HPL-MxP_260GPU.dat` | HPL-MxP inputs, repo formula, NB 2048; `mix_common.sh` turns N, NB, P, Q and PMAP into `hpl-mxp.sh` flags and adds `--fill-device 1` and the per-rank `--gpu-affinity`, as in `hplmxp_240A100.sh` |
| `dats/HPCG_256GPU.dat`, `dats/HPCG_260GPU.dat` | HPCG inputs, 1800 s; 256 uses the 512x512x256 local grid of the 61.37 TF 240-A100 run, 260 is still 256^3 |
| `alt/build_hpl_nvblas.sh` | last-resort build (section 7) |

Your tracked `dats/256GPU_grid.dat` is a 32-run sweep template and was left untouched.

## 3. How the hetjob works

An A100 node runs 4 tasks, an 8-way H100 node runs 8. A single `sbatch` job cannot vary
tasks per node, so each script is a Slurm heterogeneous job: component 0 is the 60 A100
nodes, component 1 the two 8-way nodes, component 2 (260 only) the 4-way node. Every option
after a `#SBATCH hetjob` line belongs to the next component, which is why `-p`, `-q`, `-t`
and `--reservation` are repeated. One `srun` with a colon-separated section per component
(`--het-group=0 -N 60 -n 240 ... : --het-group=1 -N 2 -n 16 ...`, geometry read from
`scontrol show job` for each component) creates one step spanning all components, so
MPI_COMM_WORLD has 256 or 260 ranks, numbered component by component: the H100 ranks are
the last 16 or 20. With column-major mapping and P=16 that puts the H100s in the last
column of the grid, which is as benign a placement as HPL offers.

Component 0 carries `-w` with the A100 list on purpose: `hpl-256` holds both node types, so
`--gpus-per-node=4` alone could land an A100 component on an 8-way node.

The per-rank `run.sh` is generated at submit time and does three things the 240 script did
not:

- **HCA.** Every rank on a node uses the HCA behind the up IPoIB interface, exactly as
  `240A100.sh` does. The first 256 run (2026-10-14) died because the script then took the first
  ACTIVE device under `/sys/class/infiniband`, which on five A100 nodes is the Ethernet port
  (`link layers differ RoCE (local) vs IB (remote)`). `HPL_HCA_LIST="mlx5_0 mlx5_1 ..."` in
  the environment overrides the choice per GPU index on every node that sees it.
- **IPoIB interface.** `NCCL_SOCKET_IFNAME` is the first `ib*` interface that is up, not a
  hard-coded `ib0`.
- **Threads.** `OMP_NUM_THREADS` follows `--cpus-per-task` of the component.

The batch environment holds component 0's `SLURM_NTASKS`, `SLURM_GPUS_PER_NODE` and
`SLURM_MEM_PER_NODE`, and srun would apply them to every section; `mix_common.sh` unsets them
and states `-N`, `-n`, `--ntasks-per-node` and `-c` per section. That is what the first 256
run on 2026-10-14 tripped over ("requested tasks 240 with the maximum number of requested
nodes 2").

Each rank prints one line starting `On node <host>:` with its GPU, HCA and PE map. Read
those lines before reading any error.

## 4. Before the first run

1. H100 nodes: `scg006` and `scg020` are 8-way, `scg016` is 4-way. Both `scg016` and
   `scg020` were drained on 13 October; resume them before the pre-flight.
2. Run the pre-flight from the controller:

       ./h100_mix_check.sh scg006 scg020 scg016

   It prints, per node: GPUs with MIG state and memory, each IB device with port state, rate
   and NUMA node, each GPU's NUMA node, the up IPoIB interfaces, whether `nvidia_peermem` is
   loaded, whether the driver carries `PeerMappingOverride=1` (IBGDA needs it; without it
   use `MIX_NVSHMEM=0`), core and memory counts, and `nvidia-smi topo -m`.
3. Set `--cpus-per-task` and `--mem` in the H100 components from the core and memory counts
   it prints. The 256 scripts carry 12 CPUs per task and 256000M, set from scg006 (96 cores)
   and scg020 (104 cores) on 2026-10-14.
4. The 256-GPU runs use the `hpl-256` partition, created 2026-10-14 with
   sg[001-050],sdg051,sg[235-239],scg[006,009-012,020] (62 nodes, 256 GPUs). `256mix.sh`,
   `mix_test12.sh` and `mix_test32.sh` submit both components to it. Confirm membership with

       sinfo -p hpl-256 -h -o '%N'

   The reservation has `ALL_NODES`, so it follows automatically. `260mix.sh` still submits to
   `hpl`, which would need scg016 added before a 260 run.
5. Run the pre-flight over that partition; its default is the `hpl` partition:

       ./hpl_preflight.sh "$(sinfo -p hpl-256 -h -o '%N')"

## 5. Validation ladder

Run these in order and do not skip a rung; each one isolates something.

| Step | Command | Proves |
|---|---|---|
| 1 | `DRY_RUN=1 bash -c 'DAT_HPL=dats/12GPU_mix_test.dat DAT_MXP=x DAT_HPCG=x source mix_common.sh'` on the login node | `run.sh` generates; shows the srun line |
| 2 | `sbatch mix_test12.sh` | hetjob launches, 12 ranks see each other, HCA mapping on the 8-way node is sane; about 5 minutes end to end |
| 3 | `MIX_NVSHMEM=0 sbatch mix_test12.sh` | the NCCL path works too, so you have a fallback |
| 4 | `sbatch mix_test32.sh` | two 8-way nodes talk to each other and to four A100 nodes with IBGDA; the per-GPU rate here should be within a few percent of the 2-node A100 rate (16 TFlop/s) |
| 5 | `sbatch 256mix.sh` | the real thing |
| 6 | `sbatch 260mix.sh` | only after 256 produced a clean number |

Reading step 4: a per-GPU rate near 16 TFlop/s means the mixed layout costs nothing and
256 should land near 2.95 PFlop/s. A rate near 5, as in May, means the H100 ranks are
funnelling through one NIC or NVSHMEM is falling back; check the `On node` lines for the
H100 nodes and set `HPL_HCA_LIST` from `nvidia-smi topo -m`.

## 6. Knobs

| Variable | Default | Effect |
|---|---|---|
| `BENCH` | `hpl` | `hpl-mxp` or `hpcg` run the other benchmarks with the matching dat file |
| `MIX_NVSHMEM` | `1` | `0` turns NVSHMEM off and nothing else, so NVSHMEM is the only variable |
| `DAT` | per script | any input file |
| `HPL_HCA_LIST` | unset | explicit GPU-index to HCA list, see section 3 |
| `SIF` | `nvhpl_26.02.01.sif` | another container release; 26.02.01 produced all three 240-A100 results on 2026-10-14 |
| `DRY_RUN` | `0` | `1` writes `mymix/run.sh` and prints the srun line only |

`NCCL_P2P_DISABLE=1` is carried over from the 240 script. On an 8-way NVSwitch node it
forces NCCL through host memory; if step 4 shows the H100 nodes slow while the HCA mapping
is right, that is the next thing to flip, in `mix_common.sh`. `BENCH=hpl-mxp` already runs with it unset,
matching the 16.15 PFlop/s 240-A100 MxP run.

## 7. The last resort

`alt/build_hpl_nvblas.sh` builds netlib HPL 2.3 against OpenBLAS with NVIDIA's NVBLAS
intercepting DGEMM. It runs on any CUDA GPU and needs none of the NVSHMEM, NCCL or UCX
machinery, which makes it a clean way to prove the fabric and the hetjob layout. It is also
3 to 5 times slower per GPU than the container, because only DGEMM is offloaded and every
call crosses PCIe, so at 260 GPUs it lands around 1 to 1.5 PFlop/s. It cannot produce the
submitted number. The script prints the run recipe when it finishes.

## 8. Submitting a mixed system

TOP500 accepts a mixed-accelerator entry. The form takes both accelerator types and counts;
Rpeak is computed from each card's full FP64 rate, so the H100s inflate Rpeak and the
efficiency column looks worse than the A100-only entry. That is cosmetic. Rmax is what ranks.
