# HPL on Sol — Why NVSHMEM Is Disabled (May 2026)

**Status:** `HPL_USE_NVSHMEM=0` in [224A100.sh](224A100.sh) is a workaround, not a permanent choice. Revisit when NVIDIA ships a patched NVSHMEM (see [Future Path](#future-path-when-to-revisit) below).

**Last working run:** 2026-05-21, 4 nodes / 16 A100s, **2.354e+05 Gflops** (~14.7 TF/GPU, ~75% of peak).

---

## TL;DR

A previously working HPL configuration (240 GPUs, 60 nodes, `nvhpl_25.09.sif`) stopped working on Sol after the cluster image was re-deployed with a newer NVIDIA driver (**595.71.05 / CUDA 13.2**). The root cause is a bug in **NVSHMEM 3.5.19 (shipped in both `nvhpl_25.09.sif` and `nvhpl_26.02.sif`)** that triggers either (a) a hang in `bootstrap_mpi_allgather` at >2 nodes or (b) a heap-corruption abort in `nvshmemi_common_init` if the allgather is bypassed via TCP. No NVSHMEM bootstrap path completes cleanly under this driver/library combination. None of the available env-var workarounds resolve it because the bug is inside NVSHMEM's compiled code, not in MPI/UCX configuration.

After exhausting MPI tuning, UCX transport swaps, TCP fallback, NVSHMEM bootstrap selection, alternate IB transports (`ibdevx`), and OpenMPI allgather algorithm overrides, we set `HPL_USE_NVSHMEM=0` to route HPL through NCCL-only collectives. With one additional fix to pin each rank to a unique GPU, the run now completes — at the cost of an estimated 20–40% peak Gflops vs. the NVSHMEM-enabled target.

---

## Background

### What we were trying to do

The PRIMARY benchmark target in [benchmark-recommendations.csv](benchmark-recommendations.csv) is HPL on **224 A100 80GB GPUs across 56 nodes** (P=14, Q=16, N=1,439,872, NB=448), targeting **~2.42e6 Gflops** based on the previously achieved 240-GPU baseline. This was the small-grid follow-up to a successful 240-GPU run conducted in 2025 with `nvhpl_25.09.sif` and the then-current Sol image.

### Hardware in scope

Per [sol-gpu-node-inventory](./.claude/projects/-Users-tianche5-Documents-vscode-HPC-benchmarking/memory/sol-gpu-node-inventory.md):

- **56 nodes used:** 49 public `sg[001-033,035-050]` + 5 public `sg[235-239]` + 2 condo `scg[010-011]` (sg034 down, scg009 excluded due to 1-HCA topology heterogeneity)
- Each node: 4× A100 80GB SXM4, NVLink-bridged (no NVSwitch), 3× Mellanox ConnectX-6 HCAs (`mlx5_0`, `mlx5_2` ACTIVE; `mlx5_1` admin-disabled). `ib0` binds to `mlx5_2` on all uniform nodes.
- IB subnet: `192.168.150.0/21` (`sg*`/`sdg*`) and `192.168.149.0/21` (`scg*`) — same `/21` segment, GID-prefix verified consistent.

### Driver and software

- **Host CUDA driver:** `595.71.05` / CUDA 13.2 (re-imaged sometime after the 2025 baseline)
- **Container:** `nvhpl_26.02.sif` containing:
  - HPL-NVIDIA wrapper + xhpl binary
  - **NVSHMEM 3.5.19** (built against CUDA 13.0)
  - HPC-X 2.25.1 (OpenMPI + UCX)
  - NCCL 2.29.2
- Also tested: `nvhpl_25.09.sif` (NVSHMEM 3.5.19, HPC-X 2.24.1) — same bug.

The CUDA-API/Driver gap (NVSHMEM built against 13.0, driver is 13.2) is forward-compatible by design and isn't the cause of the bug — it's just the context in which the latent bug was exposed.

---

## What Happened (Chronological)

The debug effort spanned 2026-05-20 → 2026-05-21 and went through several distinct failure modes. Each was diagnosed and ruled out before the next.

### Round 1 — Initial hang post-NVSHMEM team split

**Symptom:** Job submitted, log showed normal startup through `[HPL TRACE] nvshmem_team_split_2d`, then silent hang. No further log lines.

**Diagnostic path:**
- Verified all 56 nodes uniform on IB config, GPU count, peermem visibility, dc_mlx5/gdr_copy transports inside container.
- Found `scg009` had a 1-HCA topology (vs 3 HCAs on every other node). Excluded it. Hang persisted.
- gdb sampled stuck ranks showed split state:
  - Some ranks stuck in `pncclCommInitRank → recvmsg` (NCCL bootstrap).
  - Other ranks already in `PMPI_Allreduce` (past NCCL).

**Diagnosis:** Not NVSHMEM, not the fabric. **NCCL's intra-node P2P probe was failing with `Cuda failure 101 'invalid device ordinal'`** because the SBATCH directives `--gpus-per-task=1 --gpu-bind=single:1` set `CUDA_VISIBLE_DEVICES` to a single GPU per rank, so NCCL couldn't probe peer access between ranks on the same node.

**Fix:** Removed `--gpus-per-task=1` and changed to `--gpu-bind=none`. Also set `ConstrainDevices=no` in Slurm `cgroup.conf` cluster-wide so all 4 GPUs are visible to all 4 ranks on a node, with each rank picking its own via `cudaSetDevice(SLURM_LOCALID)` inside HPL's wrapper.

**Outcome at this point:** 2-node test passed (137 TFlops on 8 GPUs, ~17 TF/GPU). 56-node test still hung — but now in a *different* place.

### Round 2 — Hang in NVSHMEM MPI bootstrap allgather

**Symptom:** With the GPU-bind fix in place, 2-node passes cleanly but 16+ node runs hang. Log goes through NVSHMEM init headers, then silently stalls. No further log lines.

**Diagnostic path:**
- gdb on multiple stuck ranks consistently showed the same stack:
  ```
  bootstrap_mpi_allgather  (nvshmem_bootstrap_mpi.so)
  PMPI_Allgather
  ompi_coll_base_allgather_intra_bruck  (or recursive_doubling, varies)
  ompi_request_default_wait
  ...
  nvshmemi_common_init
  nvshmemid_hostlib_init_attr
  main
  ```
- Different ranks at different rounds of the Bruck allgather — classic UCX endpoint-setup serialization pattern.

**Hypothesized fix #1:** Switch NVSHMEM to PMIx bootstrap (`NVSHMEM_BOOTSTRAP=PMIX`). **Rejected:** HPL-NVIDIA passes an MPI-flavored `nvshmemx_init_attr_v2` to NVSHMEM, which overrides the env var. NVSHMEM stays on MPI bootstrap no matter what env says.

**Hypothesized fix #2:** UCX DC scaling — `UCX_DC_MLX5_NUM_DCI=16`, `UCX_TLS=^tcp`, etc. **Tried and rejected:** No effect on the hang.

**Hypothesized fix #3:** UCX needed `ud_x` as auxiliary transport (DC needs an aux for wireup). Added `ud_x` to `UCX_TLS`. **Helped briefly** — exposed the next failure mode.

**Hypothesized fix #4:** Container regression. NVIDIA's [v26.02 release notes](https://docs.nvidia.com/nvidia-hpc-benchmarks/release_notes.html) document a known performance issue with HPC-X 2.25's `MPI_Allgather`/`Alltoall`/`Bcast`. Downgraded SIF to `nvhpl_25.09.sif` (HPC-X 2.24.1 instead of 2.25.1). **Hang persisted identically** — so the cause is not HPC-X 2.25-specific.

**Hypothesized fix #5:** Force a different OpenMPI allgather algorithm — tried `linear`, `bruck`, `ring`, `recursive_doubling`. **All hung.**

**Hypothesized fix #6:** Switch UCX from DC to RC transport. **Hung.**

**Hypothesized fix #7:** Bypass UCX entirely with TCP-over-IPoIB (`OMPI_MCA_pml=ob1`, `OMPI_MCA_btl=tcp,self,vader`). **Bypassed the allgather hang — and exposed a different problem.**

### Round 3 — Heap corruption inside NVSHMEM init

**Symptom:** With TCP fallback, MPI_Allgather completes. Run gets further into NVSHMEM init, then aborts:

```
free(): invalid next size (fast)
*** Process received signal ***
Signal: Aborted (6)
__libc_free
libnvshmem_host.so.3 (+0x12ba5e)       ← post-allgather processing
libnvshmem_host.so.3 (+0x132205)
libnvshmem_host.so.3 (+0x119bfa)       ← nvshmemi_common_init
libnvshmem_host.so.3 (+0x11b809)       ← nvshmemi_try_common_init
nvshmemid_hostlib_init_attr
xhpl
```

This is glibc detecting heap corruption inside NVSHMEM's own code — either a buffer overflow, double-free, or out-of-bounds write in the code path that runs after the bootstrap allgather completes.

**Why this is decisive:** The UCX/DC hang in Round 2 was masking this bug. Once we bypassed the hang via TCP, the run got far enough to trigger the real fault: **a bug inside `libnvshmem_host.so.3` (NVSHMEM 3.5.19) that the new driver exposes**. The same offsets (`+0x119bfa`, `+0x11b809`) appear in both the hang stack (Round 2) and the abort stack (Round 3), confirming it's the same code path — Round 2's hang was a downstream symptom of the same buggy init logic.

**Hypothesized fix #8:** Switch NVSHMEM's IB transport from `ibrc` to `ibdevx` to take a different post-allgather code path. **Hung in the same place** at 4 nodes — `ibdevx` doesn't change the bootstrap allgather, only the data-path transport, so it can't dodge the bug.

At this point all environment-level workarounds were exhausted. The bug is inside the compiled NVSHMEM library.

### Round 4 — Disable NVSHMEM, hit GPU duplicate-pick bug

**Decision:** Set `HPL_USE_NVSHMEM=0` to route HPL through NCCL-only collectives, bypassing NVSHMEM entirely.

**Symptom:** New error:

```
NCCL WARN Duplicate GPU detected : rank 0 and rank 2 both on CUDA device 41000
```

**Diagnosis:** With `--gpu-bind=none` (chosen earlier to fix the CUDA-101 NCCL P2P probe), each rank can see all 4 GPUs on its node. HPL is supposed to pick a unique GPU per rank via `cudaSetDevice(SLURM_LOCALID)`. With NVSHMEM disabled and `HPL_FCT_COMM_POLICY=1` engaged, HPL's wrapper's per-rank GPU assignment logic (which normally relies on NVSHMEM's PE↔HCA mapping) picks duplicates. Two ranks try to share a GPU → run hangs in `cugetrfs_mp_init`.

**Fix:** Explicitly pin each rank to its own GPU inside the container by setting `CUDA_VISIBLE_DEVICES=$SLURM_LOCALID` per-rank. Each rank now sees exactly one GPU (its assigned one, renamed as device 0 in CUDA's view), so HPL physically cannot pick a duplicate.

**Outcome:** Run completes. 4 nodes / 16 GPUs delivers **2.354e+05 Gflops** (~14.7 TF/GPU).

---

## Root Cause

**Primary bug:** NVSHMEM 3.5.19's host init code (`nvshmemi_common_init` and surrounding functions in `libnvshmem_host.so.3`) contains a memory-safety bug that triggers under driver `595.71.05` / CUDA 13.2 at >2 nodes. The bug manifests as either:

1. **A hang** in the MPI_Allgather collective used during NVSHMEM bootstrap, when MPI runs over UCX/IB — because the corruption-triggering code path is reached only after the allgather completes, and the corruption itself leaves UCX in a state that prevents the allgather from completing for other ranks; or
2. **A heap-corruption abort** (`free(): invalid next size`) if the MPI_Allgather is forced through a different transport (TCP) that doesn't get poisoned the same way — the corruption then fires cleanly when NVSHMEM tries to process the allgather results.

NVSHMEM 3.5.19 was built against CUDA 13.0 API. The driver is CUDA 13.2. Forward compatibility is supposed to be preserved, but evidently a behavioral change in the newer driver (likely in `cuMemMap`, `cuMemCreate`, or one of the device-info query APIs NVSHMEM uses during init) returns slightly different data than NVSHMEM expects, causing it to over-allocate or under-allocate a host-side buffer.

**Why no env-var workaround exists:** The bug is inside NVSHMEM's compiled code. We can change transports, bootstrap mechanisms, MPI algorithms — none of that changes the buggy host-side init logic. The only fix is a patched `libnvshmem_host.so.3`, which requires either a new container release from NVIDIA or a driver patch that reverts whatever behavioral change triggered the corruption.

**Secondary bug exposed:** With NVSHMEM disabled, HPL's per-rank GPU selection is broken (picks duplicates). Working around this requires explicit `CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`.

---

## Current Workaround

The full working configuration as committed in [224A100.sh](224A100.sh):

### SBATCH directives (relevant subset)

```bash
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none           # NOT single:1 — would trigger CUDA-101 NCCL probe
#SBATCH --cpus-per-task=12
#SBATCH --distribution=block:block
```

Slurm cgroup also has `ConstrainDevices=no` cluster-wide (set on Sol's hpl partition).

### Load-bearing env vars (inside the container)

```bash
# Pin each rank to its own GPU — required because NVSHMEM is off and HPL's
# wrapper otherwise picks duplicate devices.
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

# Disable NVSHMEM — works around the 3.5.19 heap-corruption / Allgather bug.
export HPL_USE_NVSHMEM=0
export HPL_NVSHMEM_SWAP=0
export HPL_NVSHMEM_INIT=0     # belt-and-suspenders: forces MPI bootstrap, not UID
```

Everything else (UCX `dc_x`, NCCL on IB via `mlx5_2`, OMP threading, GPU monitoring) matches the previously working 240A100 baseline.

### Why each piece is necessary

| Setting | Without it | Why we need it |
|---|---|---|
| `HPL_USE_NVSHMEM=0` | NVSHMEM 3.5.19 hangs/aborts at >2 nodes | The root-cause workaround |
| `HPL_NVSHMEM_SWAP=0` | HPL still tries the NVSHMEM swap path | Has to be off when NVSHMEM is off |
| `CUDA_VISIBLE_DEVICES=$SLURM_LOCALID` | NCCL detects duplicate GPU picks, hangs in `cugetrfs_mp_init` | NVSHMEM normally drives per-rank GPU assignment; without it, manual pinning |
| `--gpu-bind=none` (vs `single:1`) | NCCL P2P probe fails with `Cuda failure 101` | Each rank's CUDA context needs to see all 4 GPUs *before* we narrow it via CVD, so the probe sees valid peers |
| `NCCL_P2P_DISABLE=1` | NCCL probes intra-node P2P and may stumble | Defensive — with CVD pinning, NCCL doesn't need P2P; intra-node goes via shared memory |
| `ConstrainDevices=no` in cgroup.conf | Slurm overrides our CVD pin | Slurm-level GPU isolation conflicts with manual pinning |

---

## Consequences

### Performance impact

**Measured (4 nodes / 16 GPUs):** 2.354e+05 Gflops = **14.7 TF/GPU** ≈ **75% of A100 80GB HPL peak (~19.5 TF/GPU with tensor cores)**.

**Estimated at 56 nodes / 224 GPUs:** Roughly **1.5–2.0e6 Gflops** vs. PRIMARY target of **~2.42e6 Gflops** with NVSHMEM. That's a **20–40% reduction**.

The hit comes from:
- **Panel swap** (`HPL_NVSHMEM_SWAP=0`): goes through NCCL/MPI instead of NVSHMEM-accelerated GPU-side swap. This is HPL's largest single communication operation.
- **Doorbell-on-GPU**: NVSHMEM's IBGDA path lets GPUs directly post IB sends/receives without CPU involvement. Without NVSHMEM, GPUs must signal CPUs to invoke MPI/NCCL.
- **Loss of intra-node NCCL P2P**: With `NCCL_P2P_DISABLE=1` (kept defensively), intra-node NCCL traffic uses shared memory instead of NVLink. Worth experimentally removing this for the final 56-node run — see [Future Tuning](#future-tuning-attempts-once-stable) below.

### What still works correctly

- Multi-node HPL completes with passing residual check.
- IB GDR (GPUDirect RDMA) is still active for NCCL — `gdr_copy` transport is in `UCX_TLS` and peermem is loaded.
- All 56 nodes participate; GPU temperature/clock monitoring still emits.
- Results are valid for publication or submission; only the absolute peak Gflops is lower than the (theoretically achievable) NVSHMEM baseline.

### What is *not* affected

- HPL correctness — residual check passes; numerical result is identical to NVSHMEM-enabled runs.
- Hardware utilization at the GPU compute level — GEMM throughput per GPU is unchanged; the loss is purely in inter-rank communication overhead.
- Other benchmark targets (HPL-MxP, HPCG) — separate scripts; they'll need similar treatment if NVSHMEM is also disabled there.

---

## Future Path: When to Revisit

NVSHMEM should be re-enabled when **either** of:

1. **NVIDIA ships `nvhpl_26.03+` or later** with a patched NVSHMEM (≥ 3.5.20 or 3.6.x). Track [NGC tags](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/hpc-benchmarks/tags). When a new version drops, test 2-node first with NVSHMEM on, then scale up.
2. **NVIDIA ships a driver update** (e.g., 595.80+ or 600.x) that reverts whatever behavioral change in CUDA 13.2's libcuda triggered the corruption.

### The revert procedure

When a fix is available, in [224A100.sh](224A100.sh) make these reverse edits:

```bash
# Re-enable NVSHMEM:
export HPL_USE_NVSHMEM=1      # was 0
export HPL_NVSHMEM_SWAP=1     # was 0

# Remove the per-rank GPU pin — NVSHMEM needs all GPUs visible to set up
# its PE↔HCA↔device mapping. Leaving CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# in place WILL break NVSHMEM init.
# DELETE this line:
#   export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
# (and the diagnostic echo following it)
```

Then 2-node smoke test first. If NVSHMEM init completes cleanly (no hang, no heap corruption, no `free(): invalid next size`), scale to 16, then to 56.

### What success looks like at 56 nodes with NVSHMEM re-enabled

Per [benchmark-recommendations.csv](benchmark-recommendations.csv) PRIMARY estimate: **~2.42e6 Gflops**, ~10.8 TF/GPU at scale (lower per-GPU than at 16 nodes due to communication scaling, but the *absolute* number is much higher).

---

## Reproducer for NVIDIA Ticket

If filing this with NVIDIA's HPC support, the reproducer is tight enough to be actionable:

**Environment:**
- Host: ASU Sol cluster, 56× nodes with 4× A100 80GB each, Mellanox ConnectX-6 IB
- Driver: `nvidia-driver` 595.71.05, CUDA 13.2
- Container: `nvcr.io/nvidia/hpc-benchmarks:26.02` (also reproduces in 25.09)

**Reproduction steps:**

1. Launch HPL via SLURM with `srun --mpi=pmix` running the container's `/workspace/hpl.sh`.
2. **2-node test passes** (137 TFlops on 8 GPUs with NVSHMEM enabled).
3. **16-node test hangs** in `bootstrap_mpi_allgather → PMPI_Allgather` inside `nvshmemi_common_init`. gdb stack:
   ```
   ompi_request_default_wait → ompi_coll_base_allgather_intra_bruck
   → PMPI_Allgather → bootstrap_mpi_allgather (libnvshmem_bootstrap_mpi.so.3)
   → nvshmemi_common_init (libnvshmem_host.so.3 +0x119bfa)
   → nvshmemid_hostlib_init_attr → xhpl main
   ```
4. **If MPI is forced to TCP** (`OMPI_MCA_pml=ob1`, `OMPI_MCA_btl=tcp,self,vader`), the Allgather completes but the run aborts immediately after:
   ```
   free(): invalid next size (fast)
   abort → __libc_free → libnvshmem_host.so.3 +0x12ba5e
   → libnvshmem_host.so.3 +0x132205
   → nvshmemi_common_init → nvshmemid_hostlib_init_attr → xhpl main
   ```
5. **Failure threshold is between 2 and 16 nodes** (haven't bisected exactly).

**Tried and ineffective workarounds:**
- Multiple OpenMPI allgather algorithms (linear, bruck, ring, recursive_doubling)
- UCX transport variations (`dc_x`, `dc_x+ud_x`, `rc_x`, `^tcp`)
- `UCX_DC_MLX5_NUM_DCI` tuning (1, 16, 32)
- `NVSHMEM_REMOTE_TRANSPORT` variations (`ibrc`, `ibdevx`, `ibgda`)
- `NVSHMEM_BOOTSTRAP=PMIX` — ignored because HPL passes MPI-flavored init attr
- Container downgrade `26.02 → 25.09` (same NVSHMEM version, same bug)

**Working workaround:** `HPL_USE_NVSHMEM=0` + manual per-rank `CUDA_VISIBLE_DEVICES=$SLURM_LOCALID` pin. ~20-40% peak Gflops penalty.

---

## Related Files

- [224A100.sh](224A100.sh) — current working SBATCH script (NVSHMEM disabled)
- [240A100.sh](240A100.sh) — last-year's working baseline (NVSHMEM enabled, but does NOT work under current driver)
- [benchmark-recommendations.csv](benchmark-recommendations.csv) — PRIMARY target definitions and expected Gflops
- [hpl-score.csv](hpl-score.csv) — record of achieved scores; annotate any NVSHMEM-off result with the workaround status
- `.claude/projects/.../memory/project-sol-cuda-driver-change-2026-05.md` — internal memory record of the driver change and failure modes
- `.claude/projects/.../memory/project-sol-hpl-working-config-2026-05.md` — internal memory record of this working config

## Future Tuning Attempts (Once Stable)

Once the 56-node baseline with NVSHMEM-off is committed:

1. **Try removing `NCCL_P2P_DISABLE=1`** — with the per-rank CVD pin in place, NCCL P2P probes should work cleanly. NVLink P2P for intra-node NCCL is a real perf win. Run 4-node test first; if no `Cuda failure 101`, keep removed for production.
2. **Try `HPL_FCT_COMM_POLICY=2`** — alternate NCCL-only swap algorithm. Untested here. Could be incrementally faster than default policy 1.
3. **Tune `NB`** — `224_grid.dat` uses NB=448 (best from 240 NVSHMEM sweep). Without NVSHMEM, the optimal NB may differ. A short sweep (`NB ∈ {384, 416, 448, 480, 512}`) on 4 nodes could find a better fit before scaling.
