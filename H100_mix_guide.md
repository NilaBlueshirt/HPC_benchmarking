# Mixed A100 + H100 HPL on Sol

What the October 2026 window taught us about adding the H100 nodes to the 240-A100 HPL run,
and the one layout that can still be tried. Written for the Sol admin.

## 1. Findings (14 October 2026)

- **NVSHMEM requires the same number of PEs on every node.** The second 256-GPU attempt died in
  NVSHMEM's bootstrap with `NVSHMEM requires the same number of PEs on all nodes (4 PEs on sg017)`.
  The check runs before anything else and no environment setting changes it. A job with 4 PEs on
  the A100 nodes and 8 on the 8-way H100 nodes can never run with NVSHMEM. That closes the
  256-GPU layout (`256mix.sh`) and the 260-GPU one (`260mix.sh`).
- **The alternatives are all too slow to make the list:** NVSHMEM off cost 12.5% at 224 GPUs
  (2.232 against 2.552 PFlop/s), more than the 6.7% that 16 H100s could add, other container
  releases carry the same NVSHMEM rule, and netlib HPL with NVBLAS runs 3 to 5 times slower per GPU.
- **The HCA must be the device behind ib0.** The first 256-GPU attempt died in UCX with
  `link layers differ RoCE (local) vs IB (remote)`. On scg010, scg011, sdg051, sg004 and sg012 the
  device named mlx5_0 is the Ethernet port, which the mlx5 driver also lists under
  `/sys/class/infiniband` with its port ACTIVE. The mixed script used to take the first ACTIVE
  device; it now takes the device behind the up IPoIB interface, exactly as `240A100.sh` does.
- **HPL-NVIDIA 26.02.01 aborts on mixed GPU models.** The 248-GPU run (plain job, 4 PEs on every
  node) got past MPI, the HCA and NVSHMEM's PE check, printed its device info, and then died in
  `MPI_Allgather` on a copy of the world communicator with `MPI_ERR_TRUNCATE: message truncated`.
  Every rank that reported it was an A100 rank, so the H100 ranks sent more bytes than the A100
  ranks had room for: a per-rank size that depends on the GPU model (the H100 has 132 SMs to the
  A100's 108). No environment variable controls that exchange. The only lever is the container
  release: the May mixed pairs ran at full speed on an earlier release than 26.02.01.
- **Slurm orders the nodes by name.** In the plain 248 job scg006 came first, so rank 0 was an
  H100 and the H100 ranks were 0 to 3 and 20 to 23, not the last process column.
- **The heterogeneous job itself worked.** With the HCA fixed, the 256 ranks came up as one MPI
  world and reached NVSHMEM init. The hetjob mechanics are not the problem.
- **Only 4 GPUs per node mixes.** That gives 252 with scg016 (14x18 grid) or 248 without it
  (8x31). scg016 is not available in this window, so the try is 248.
- **248 is a weak number.** Its only grid is 8x31, a 1:4 rectangle against the 15x16 of the 240 run.
  Per-GPU traffic per step rises about 22% (it scales with 1/P + 1/Q) and the panel column holds
  twice the rows. The GPU-count ceiling is +3.3% over 240; the grid penalty is expected to be of the
  same size. It is a try, not a plan. 244 (4x61) is worse.
- **Earlier evidence.** The 8-GPU mixed pairs (one A100 node plus half of scg006, 4 PEs on each
  node) ran at full A100 speed, 16.2 to 16.6 TFlop/s per GPU, with the normal transport. The May
  252-GPU collapse (4.5 TFlop/s per GPU) ran with NVSHMEM off, `rc_x` and no CUDA IPC, so it says
  nothing about the wire.
- **scg006 and scg020.** Each has 8x H100 80GB HBM3 on NVSwitch and one HDR 200G HCA (mlx5_0, ib0)
  on NUMA 1. GPUs 0-3 sit on NUMA 0, GPUs 4-7 on NUMA 1 next to the NIC. An H100 80GB exposes
  81559 MiB, 361 MiB less than an A100 80GB, so mixed inputs are sized 0.5% under the 240 fill.

## 2. Files

| File | Purpose |
|---|---|
| `248mix.sh` | 240 A100 + 8 H100 (4 on each 8-way node), one plain job, 248 ranks, 8x31 |
| `dats/248GPU_mix.dat` | N 1571328 (the 2.815 PF run's fill scaled to 248 GPUs, trimmed for the H100), NB 384 twice (warm-up and run), 8x31 column-major, the algorithm line of `HPL_240GPU.dat` |
| `mix_common.sh` | shared body: builds the per-rank `run.sh` and launches `srun`; HCA from ib0 |
| `h100_mix_check.sh` | pre-flight for the H100 nodes: GPUs, IB devices, NUMA, peermem, topology |
| `hpl_preflight.sh` | pre-flight over a node list |

Kept in the repo but not runnable with NVSHMEM because they put 8 PEs on the 8-way nodes:
`256mix.sh`, `260mix.sh`, `mix_test12.sh`, `mix_test32.sh` and their inputs under `dats/`.

## 3. How 248mix.sh works

It is `240A100.sh` with two more nodes. Every node, A100 or H100, gets 4 tasks, 4 GPUs, one GPU
per task and 12 CPUs per task, so there is no heterogeneous job, no per-component geometry and
NVSHMEM's rule is met. The node list is explicit (`-w`) so Slurm cannot substitute a node, and the
job goes to the `hpl-256` partition, which holds exactly these 62 nodes.

On scg006 and scg020 Slurm picks 4 of the 8 GPUs. Which ones matters, because GPUs 4-7 share
the NIC's NUMA node and GPUs 0-3 do not. Each rank prints one `On node` line with its HCA and the
PCI bus id of the GPU it got; read the 8 lines from the H100 nodes before reading anything else.

Slurm numbers the ranks in node-name order, and `scg006` sorts before `sg001`, so the H100 ranks
are 0 to 3 (scg006) and 20 to 23 (scg020): column 0 rows 0 to 3 and column 2 rows 4 to 7 of the
8x31 grid, not a column of their own.

## 4. Before the run

1. scg006 and scg020 resumed and in `hpl-256`:

       sinfo -p hpl-256 -h -o '%N'

   Nodes inside the maintenance reservation show state `maint`; that is the ready state.
2. Clock locks as for the 240 run (`wwctl ssh sg[001-050] nvidia-smi --lock-gpu-clocks=1410`,
   1275 on the throttling nodes). The H100 nodes need no lock for this try.
3. Optional pre-flight: `./h100_mix_check.sh scg006 scg020`.

## 5. Run and read

    sbatch 248mix.sh

In `mix248.log`, in order:

1. The eight `On node scg006` and `On node scg020` lines: `gpus_on_node=4`, `hca=mlx5_0`, and
   the `bus=` ids, which tell you whether the GPUs came from NUMA 0 or NUMA 1.
2. The HPL parameter banner: N 1571328, NB 384, P 8, Q 31.
3. The `WC0` lines. The first is the warm-up, the second is the number. The ceiling is about
   2.9 PFlop/s. Anything under 2.815 means the 240 run is the submission.

The job carries 4 hours; two runs take about 35 minutes.

## 6. Knobs

Knobs are passed as arguments, `sbatch 248mix.sh KEY=VALUE`. The header has `--export=NONE`, so a
`VAR=x` prefix on the sbatch line never reaches the job; the older scripts documented that form,
and it silently did nothing.

| Knob | Default | Effect |
|---|---|---|
| `DAT` | `dats/248GPU_mix.dat` | another HPL input (P x Q must be 248) |
| `BENCH` | `hpl` | `hpcg` runs HPCG on the same 248 ranks; `hpl-mxp` would need an 8x31 MxP input, which is not written |
| `HPL_HCA_LIST` | unset | HCA names in GPU-index order, overrides the ib0 choice on every node that sees it |
| `SIF` | `nvhpl_26.02.01.sif` | another container release; 26.02.01 produced all 240-A100 results |
| `DRY_RUN` | `0` | `1` writes `mymix/run.sh` and prints the srun line without launching (run on the login node as `DRY_RUN=1 SLURM_NTASKS=248 bash -c 'DAT_HPL=dats/248GPU_mix.dat DAT_MXP=x DAT_HPCG=x source mix_common.sh'`) |

`MIX_NVSHMEM=0` still exists in `mix_common.sh` but is not an option: see section 1.

## 7. Submitting a mixed system

TOP500 accepts a mixed-accelerator entry. The form takes both accelerator types and counts;
Rpeak is computed from each card's full FP64 rate, so the H100s inflate Rpeak and the
efficiency column looks worse than the A100-only entry. That is cosmetic. Rmax is what ranks.
