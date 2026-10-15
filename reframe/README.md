# Sol pre-release checks (ReFrame)

Run this at the end of a maintenance window, before Sol goes back to production. One command, `./run_prerelease.sh`:

1. reads Slurm and picks the nodes to test (inventory),
2. submits one whole-node job to every testable node (node stage),
3. runs MPI jobs on each node group in `MPI_GROUPS`, 13 per group with the three default MPIs (MPI stage),
4. prints the nodes that need fixing, with the reason for each, and writes `report.html` with the cluster-wide numbers.

Every run is kept in `results/<run>/`, so the next maintenance can be compared with this one.

## What it checks

**Inventory** (`inventory.py`). A node is tested when it is:
- not in `EXCLUDE_NODES` (the A100 nodes validated by HPL),
- in a usable Slurm state: MAINT, RESERVED and IDLE are fine; DOWN, DRAIN, FAIL, NOT_RESPONDING, POWER*, FUTURE, REBOOT are not,
- in `RESERVATION`,
- in the `--only` list, when one is given,
- in a submit partition whose `MaxCPUsPerNode` and `MaxTime` allow a whole-node job of `NODE_TIME_LIMIT`.

Each node goes to the first `SUBMIT_PARTITIONS` entry that has it and fits. The inventory prints `testable | excluded | unavailable | untestable` with the reason for every node it skips. It warns when the reservation is not active or ends within 3 hours.

| Inventory status | In the report |
|---|---|
| unavailable (Slurm state, not in the reservation) | UNAVAILABLE, exit code 1 |
| untestable (no partition can take a whole-node job, e.g. lightwork's 8-CPU cap) | NOT_MEASURED |
| excluded | counted only |
| outside `--only` | counted as "not selected" only |

**Node stage.** One exclusive job per testable node (`tests/node_checks.py` runs `tests/src/node_check.sh`), all at once. A job takes a few minutes, with a limit of `NODE_TIME_LIMIT`. Every step has its own timeout, and a step that fails or hangs does not stop the next one.

| Step | What it does | Flagged when |
|---|---|---|
| ident | hostname, CPU architecture, kernel, slurmd version, NVIDIA driver | the job ran on the wrong node or architecture; kernel or slurmd differs from what most nodes of the same architecture report to Slurm, or the driver differs from most NVIDIA nodes (MISMATCH) |
| devices | counts accelerators: `nvidia-smi -L` (a MIG slice counts as one), AMD KFD, `hl-smi` for Gaudi | the count differs from the node's Slurm gres |
| srun | `srun -n <cores> hostname` inside the job | not every task starts, or tasks land on another host |
| mpi_hello | builds with `MOD_NODE_MPI`, then one rank per core: allreduce check, MPI_Init time, barrier time | build or start-up fails, wrong rank count or result; MPI_Init above 2x the group median (OUTLIER) |
| stream | STREAM on all cores, built with `MOD_GCC` (system gcc on aarch64) | validation fails; triad below 90% of the group median (OUTLIER) |
| gpu | NVIDIA only, `MOD_CUDA`: PCIe host↔device and memory bandwidth on each GPU or MIG slice, one at a time; then FP32 and FP64 GEMM on all devices at once for `GPU_SECONDS` each, every result compared bit for bit | a device is missing, fails or gives a different result; a value below 90% of the median for that GPU type (OUTLIER) |

CPU nodes are compared within their CPU group: CPU model, cores and memory, e.g. `zen4-128c-512G`. GPUs are pooled by name and memory.

**MPI stage** (`tests/mpi_checks.py`). Runs on the groups in `MPI_GROUPS`. Nodes with JOB_FAILED or WRONG_RESULT in the node stage are left out (listed in `results/<run>/skip_mpi.txt`). MISMATCH, TOO_SLOW and OUTLIER nodes stay in. A group with fewer than 2 usable nodes is skipped and reported as NOT_MEASURED.

| Test | Jobs per group | What it does |
|---|---|---|
| pairwise (`tests/src/pairbw.c`, `MOD_NODE_MPI`) | 1 | one rank per node on all usable nodes: bidirectional bandwidth of every node pair, one pair at a time so no two pairs share the fabric, plus latency. About 3 minutes for the 107 cpu-zen3 nodes at 200 Gb/s, about 6 at 100 Gb/s. |
| OSU pt2pt | 2 per MPI | `osu_bw` and `osu_latency` between the first two usable nodes, for each MPI in `MPI_CANDIDATES` |
| OSU collective | 2 per MPI | `osu_allreduce` and `osu_alltoall` on the group's first N usable nodes (N is the 4th field of its `MPI_GROUPS` line), one rank per core, for each MPI |
| GPU-direct, only with `GPU_DIRECT=yes` | 2 per GPU group | hpcx `osu_bw` and `osu_latency` between GPU buffers on two nodes |

The OSU tests wait for the login node to download OSU micro-benchmarks 7.5.2 and build it once per MPI. A group's jobs share its first nodes, so they run one after another; groups run in parallel. Each job may run for `MPI_TIME_LIMIT` and wait up to `MPI_MAX_PENDING` in the queue.

**Not covered.**
- IB port state and rate, mounts, memory and services: NHC already checks them.
- The A100 nodes in `EXCLUDE_NODES`: the HPL runs validated them.
- Software modules are not swept: each step loads one module, the version set in `site.conf`.

## One-time setup

1. **Install ReFrame 4.10.4.** It needs Python 3.10 or newer. The wrapper prints the same hint:
   ```
   python3.12 -m venv ~/rfm-4.10.4
   ~/rfm-4.10.4/bin/pip install reframe-hpc==4.10.4
   ```
   Then set `REFRAME="$HOME/rfm-4.10.4/bin/reframe"` in `site.conf`. Use `$HOME`: `~` does not expand inside quotes. `PYTHON` can stay `python3`, because `inventory.py` and `report.py` need only Python 3.8+ and the standard library.
2. **Location.** Keep this `reframe/` folder on a filesystem that the compute nodes can read and write. Jobs compile and write their output in `results/<run>/rfm/stage/`.
3. **OSU download.** If the login node has no internet, download the tarball from `OSU_URL` elsewhere and set `OSU_TARBALL` to its path.
4. **Check `site.conf`.** The file is commented. Durations need a unit: `20m` is 20 minutes, a bare `20` is 20 seconds.

| Key | Default | Meaning |
|---|---|---|
| `RESERVATION` | `oct2026_maint` | every job gets `--reservation`, and only nodes in it are tested (empty: no reservation) |
| `EXCLUDE_NODES` | `sg[001-050],sdg051,sg[235-239],scg[009-012]` | never tested |
| `SUBMIT_PARTITIONS` | `general public htc highmem lightwork fpga gaudi arm` | partitions in order of preference |
| `QOS_MAP`, `QOS` | `general=private`, empty | `--qos` per partition, as space-separated `partition=qos` pairs; partitions not in the map get `QOS` (empty: no `--qos`) |
| `ACCOUNT` | empty | `--account` for every job |
| `MAX_PENDING` | `30m` | node stage: a job still pending after this is cancelled |
| `MPI_MAX_PENDING` | `4h` | the same for the MPI stage, where a group's jobs queue behind each other |
| `NODE_TIME_LIMIT`, `MPI_TIME_LIMIT` | `20m`, `15m` | job time limits; a hung MPI job holds up its group this long |
| `MOD_GCC`, `MOD_CUDA`, `MOD_NODE_MPI` | `gcc/16.1.0`, `cuda-13.2.1-gcc-13.2.0`, `openmpi/5.0.8-gcc-15.2.0` | node-stage modules. `MOD_GCC_AARCH64`, `MOD_CUDA_AARCH64` and `MOD_NODE_MPI_AARCH64` set them for the GH200 nodes; while those are empty, STREAM uses the system gcc and the GPU and MPI hello steps are skipped |
| `MPI_CANDIDATES` | openmpi, mpich, hpcx | `key=module` for each MPI tested with OSU |
| `SRUN_MPI`, `SRUN_MPI_<key>` | `pmix` | the `srun --mpi` plugin. `<key>` is an `MPI_CANDIDATES` key, `node_mpi` (MPI hello and pairwise) or `hpcx` (GPU-direct); e.g. `SRUN_MPI_mpich="pmi2"` |
| `MPI_GROUPS` | cpu-zen3, cpu-zen4, gpu-a30, gpu-l40s | one line per group: name, partition, hostlist, nodes for collectives, gpu yes/no, separated by `\|` |
| `GPU_DIRECT` | `no` | `yes` adds the GPU-direct tests |
| `STREAM_ELEMS`, `GPU_SECONDS` | 2 GiB per array, `20` | benchmark sizes |
| `OUTLIER_FRACTION`, `HISTORY_DROP` | `0.90`, `0.10` | report thresholds |

Any other `KEY=` line is passed on too. For example, `MAX_JOBS="100"` limits how many jobs ReFrame keeps submitted at once; the default is 400.

**QOS.** ASU moves jobs submitted as general/public to public/public, so the suite submits jobs on general with `--qos=private` (`QOS_MAP`). Check that your account may use it before the full run:
```
sbatch -p general --qos=private --reservation=oct2026_maint -w scc012 -t 2 --wrap hostname
sacct -X -j <jobid> -o jobid,partition,qos,state,nodelist     # expect general, private, COMPLETED, scc012
```
The smoke test checks the same thing: scc012 and scg017 are submitted to general with QOS private. If the QOS is refused, every general node comes back JOB_FAILED with the same reason.

## Running it

```
./run_prerelease.sh --dry-run                       # job scripts only, nothing submitted
./run_prerelease.sh --only 'scc012,sc006,scg017'    # smoke test: node stage on 3 nodes
./run_prerelease.sh                                 # full run: node stage, MPI stage, report
```

| Option | Effect |
|---|---|
| `--only HOSTLIST` | test only these nodes; runs only the node stage unless `--stage` is given |
| `--stage node\|mpi\|all` | the stages to run (default `all`); `mpi` continues an existing run |
| `--run-id ID` | name of the run directory (default `YYYYMMDD-HHMM`); letters, digits, `.`, `_` and `-` |
| `--baseline ID` | compare with `results/ID` (default: `results/baseline` when it exists) |
| `--parent ID` | with `--only`: the full run to compare a small rerun with (default: the newest full run in `results/` that finished its node stage) |
| `--dry-run` | write the job scripts without submitting anything |
| `--bless ID` | make `results/ID` the baseline; takes no other option |

Start every real run on a login node inside `tmux` or `screen`.

**Dry run.** It creates `results/<run>/` with `nodes.json` and one `rfm_job.sh` per job under `rfm/stage/sol/compute/<environ>/<test>/`. To read one node's script: `grep -l 'nodelist=scg017$' results/<run>/rfm/stage/sol/compute/builtin/*/rfm_job.sh`. Nothing is submitted.

The wrapper checks the durations in `site.conf` before anything else. A value that is not a ReFrame duration stops it with exit 2. A bare number, which ReFrame reads as seconds, only gets a warning.

**Smoke test first.** Before some 220 jobs go out, run the node stage on one general node (QOS private), one public node and one GPU node: `--only 'scc012,sc006,scg017'`. In a few minutes it shows whether the reservation, QOS, modules, srun/PMIx and the CUDA build work. To cover every submit partition and accelerator type, add one node of each, for example sc005 (htc), sh001 (highmem), sfpga01i (fpga), sg250 (MI200), gaudi001 and scgh001 (GH200). To try the MPI stage on a small scale too, give two nodes of a group and `--stage all`, e.g. `--only 'sc[006-007]' --stage all`.

**Timing.** The node stage takes about 10 minutes once the jobs start. A job still pending after `MAX_PENDING` is cancelled and reported as JOB_FAILED. A job blocked for a reason such as ReqNodeNotAvail or an invalid QOS is cancelled within a few minutes. The MPI stage takes roughly 30–60 minutes: first the OSU builds, then each group's 13 jobs one after another. cpu-zen3 takes longest.

### Ctrl-C, a dropped terminal, leftover jobs

| What happened | ReFrame | Wrapper |
|---|---|---|
| Ctrl-C (press it once and wait: with ~220 jobs, cancelling takes a few minutes) | cancels each of its jobs with `scancel` and saves the stage report | prints `run_prerelease.sh: interrupted; results so far are in results/<run>` plus the `squeue` line below, and exits 130; no report, no MPI stage |
| terminal closed or SSH dropped, without tmux | the same: cancels its jobs on SIGHUP | dies without a message |
| ReFrame killed (`kill -9`, out of memory), or its session stopped on an error such as 60 failed `sacct` calls in a row | its jobs are **not** cancelled | prints `ERROR: <stage> stage: ReFrame stopped early (exit N; see results/<run>/reframe_<stage>.log)` and the two commands below, exits 3 |

In every case, check for leftover jobs before you start again. An interrupted node stage cannot be resumed, so start a new run. All suite jobs are named `rfm_Sol...`:
```
squeue -h -u $USER -o '%i %j %T' | awk '$2 ~ /^rfm_Sol/'                             # list
squeue -h -u $USER -o '%i %j' | awk '$2 ~ /^rfm_Sol/ {print $1}' | xargs -r scancel   # cancel
```

### Continuing the MPI stage

```
./run_prerelease.sh --stage mpi --run-id 20261014-2130
```
Use this when a run's node stage finished but its MPI stage did not run to the end. That covers an interrupted MPI stage and the wrapper's message `rerun it with --stage mpi --run-id <id>`. It is also the way to redo the MPI stage after changing an MPI setting in `site.conf`. It reuses that run's `nodes.json` and node results, so nodes that failed there stay out. It replaces `stage_mpi.json` and writes a new report. It cannot be combined with `--only`.

### Wrapper messages

| Message | Meaning |
|---|---|
| `WARNING: ignoring SOL_SCONTROL_NODES_FILE=...` (or another `SOL_SCONTROL_*_FILE`, `SOL_MODULES_SYSTEM`) | a setting for offline tests is left in your shell. The wrapper ignores it and uses live `scontrol` and Lmod. Only `SOL_OFFLINE=1` keeps it, and that is for testing the suite without Slurm. |
| `WARNING: modules loaded in this shell: ...` | information only: every job starts with `module purge` |
| `WARNING: written for ReFrame 4.10.4, found ...` | a different ReFrame version is installed |
| `inventory.py failed (exit N); nothing was submitted` | `scontrol` failed or no node is testable; exit 2 |
| `results/<id> already exists` | pick another `--run-id`, or use `--stage mpi` to continue that run |
| `== N node(s) failed the node stage and stay out of the MPI stage` | the nodes in `skip_mpi.txt` |
| `ERROR: <stage> stage: ReFrame skipped a test file that failed to load` | a Python error in a test file, see the log; exit 3 |
| `ERROR: the node stage did not run to the end, so the MPI stage was not started` | exit 3 |
| `MPI_MAX_PENDING="4 hours" in site.conf is not a ReFrame duration ...` | fix the value (e.g. `30m`, `4h`, `1h30m`); exit 2 |
| `== parent run <id> (newest full run in results/ ...)` | an `--only` run is compared with that full run, see [After fixing nodes](#after-fixing-nodes) |
| `== no parent run: ...` | no full run with a node stage exists yet, so a small `--only` run gets no OUTLIER or driver check |

## Reading the result

The terminal summary is what you act on (shortened):

```
Sol pre-release check  run=20261014-2130  reservation=oct2026_maint  baseline=20260415-0910
Nodes: 222 testable | 206 passed | 16 with problems | 8 not tested | 60 excluded

SYSTEMIC (shared by many nodes; the per-node rows are in problems.csv)
  [gpu] NOT_MEASURED on 4 nodes scgh[001-003],sgh001: no CUDA module configured for aarch64 (set MOD_CUDA_AARCH64 in
    site.conf)

PROBLEM NODES (worst first)
  NODE         CATEGORY              CHECK      REASON
  scc032       UNAVAILABLE           slurm      Slurm state IDLE+DRAIN+MAINT+RESERVED: Kill task failed
  scg018       WRONG_RESULT          devices    found 3 nvidia devices, Slurm gres says 4

MPI JOBS
  GROUP     MPI      TEST                 RESULT  KEY NUMBERS
  cpu-zen3  node-mpi pairwise             PASS    n=104  link_bw_min 11,140  link_bw_median 24,145 MB/s  lat_max 2.83 us
  cpu-zen4  mpich    collective/alltoall  FAIL    wrong result: OSU table stops at 512 bytes, expected 4096; stderr: ...

Files: results/20261014-2130/report.html  problems.csv  metrics.csv
Rerun problem nodes: ./run_prerelease.sh --only 'scc032,scg018'
```

"With problems" counts tested nodes with any finding other than NOT_MEASURED or SLOW_LINK. "Not tested" counts unavailable and untestable nodes. MPI JOBS has one row per group, MPI and test. A row reads PASS or FAIL; SKIP means that MPI's OSU build failed, and NOTRUN means the group was left out. MPI failures are reported per group and MPI, never per node.

Problem categories, worst first:

| Category | Meaning | Usual action |
|---|---|---|
| UNAVAILABLE | not tested: Slurm has the node DOWN, DRAIN, FAIL, NOT_RESPONDING or similar, or it is not in the reservation | fix and resume the node, or add it to `EXCLUDE_NODES` |
| JOB_FAILED | the job never ran or did not finish: sbatch error, blocked or cancelled in the queue, pending past `MAX_PENDING`, time limit (the reason names the step it stopped in) | look at Slurm and at the node |
| WRONG_RESULT | a check failed; CHECK names the step, e.g. `devices  found 3 nvidia devices, Slurm gres says 4` or `timeout  stream exceeded 600 s` | fix the hardware, driver or configuration |
| MISMATCH | the node's kernel or slurmd version differs from what most nodes of its architecture report to Slurm (`scontrol`, all nodes), or its NVIDIA driver differs from most NVIDIA nodes in this run (in a small rerun: the parent run). The reason names the source, e.g. `(from scontrol, 283 of 285 x86_64 nodes)`. | reimage or reboot the node |
| TOO_SLOW | below 90% (`OUTLIER_FRACTION`) of the reference in `references.json`, which exists only after `--bless`: STREAM triad and copy per CPU group, GPU values per GPU type (the node's slowest device) | investigate |
| OUTLIER | below 90% of its group median: STREAM triad, each GPU value, pairwise median link bandwidth per node; MPI_Init above 2x the median. A group needs at least 3 values; in a small `--only` rerun the parent run's median is used instead. | investigate |
| SLOW_LINK | one node pair below 90% of the group's pair median while neither node is an OUTLIER | check the cable or switch port |
| SLOWER_THAN_BASELINE | more than 10% (`HISTORY_DROP`) slower than the same node or GPU in the baseline run; MPI_Init and barrier only above 2x the baseline time | compare with the previous maintenance |
| NOT_MEASURED | information only: a step skipped (e.g. no aarch64 module), a node no partition can take (the lightwork cap), an MPI group with fewer than 2 usable nodes | none, or fill in `site.conf` |

A TOO_SLOW or OUTLIER row also lists that value's other findings, e.g. `also 73% of zen4-128c-512G median, -25% vs baseline`.

**SYSTEMIC.** The same finding on many nodes is printed once, as `[check] CATEGORY on N nodes HOSTLIST: reason`:
- A finding is grouped when its category, check and reason match on 5 or more nodes, or on at least half (and at least 2) of the nodes of a CPU group, GPU type or MPI group. Numbers, host names, GPU UUIDs and PCI addresses are ignored in the comparison.
- A SLOW_LINK group reads `on N links among M nodes`. If the texts differ, the reason starts with `e.g. <node>:`.
- The grouped rows are left out of PROBLEM NODES but stay in `problems.csv` and under "Systemic issues" in `report.html`. They still count for the exit code and the rerun line.

A SYSTEMIC line usually has one cause, such as a module, a `site.conf` setting, the image or a switch. Fix that first.

**Exit code.**

| Code | Meaning |
|---|---|
| 0 | no tested node has a problem from UNAVAILABLE to TOO_SLOW, and every MPI-stage test passed |
| 1 | at least one such node, a failed MPI-stage test or OSU build (the report then says which), or no node-stage results at all (the report says NOT VERIFIED) |
| 2 | setup error: options, `site.conf`, Python, ReFrame not found, inventory failed |
| 3 | a stage did not run to the end: ReFrame stopped early, a test file did not load, or the MPI stage could not start. The report is still printed. |
| 130 | interrupted with Ctrl-C |

OUTLIER, SLOW_LINK, SLOWER_THAN_BASELINE and NOT_MEASURED do not change the exit code, and neither do MPI groups left out for having fewer than 2 usable nodes (NOTRUN).

**report.html** is a single page with no external files. Add `?theme=light` or `?theme=dark` to its URL to force a theme. It has:
- the verdict, node counts, the problem table and the systemic issues;
- per-group dot charts as % of the group median, with the outlier zone shaded;
- a pairwise heatmap per MPI group: a row and column that stand out from the rest are a slow node, and a single cell pair that stands out is a slow link;
- the OSU results and curves for each MPI;
- the change against the baseline, when there is one;
- the rerun command and the thresholds used.

Every chart has a Table button that lists its numbers.

**One node's job files.** A failed job stays in `results/<run>/rfm/stage/`; a passed one is moved to `rfm/output/`. There you find `rfm_job.sh`, `rfm_job.out` with the `SOL_*` result lines, and `rfm_job.err` with each step's log. To find them:
```
grep -l 'nodelist=scg017$' results/<run>/rfm/*/sol/compute/builtin/*/rfm_job.sh
```
`reframe_node.log` and `reframe_mpi.log` in the run directory hold ReFrame's own output, including its failure details.

## After fixing nodes

```
./run_prerelease.sh --only 'scg016,scc032'      # node stage on just these nodes
```
The terminal and `report.html` print a ready `--only` line for the nodes with problems from UNAVAILABLE to OUTLIER.

A rerun of one or two nodes is still checked properly:
- **Kernel and slurmd** are compared with what all nodes report to Slurm.
- **Performance and the NVIDIA driver** are compared with the *parent run*: by default the newest full run in `results/` that finished its node stage. The wrapper prints which one (`== parent run <id> ...`); `--parent ID` picks another. A group with fewer than 3 values in the rerun uses the parent's median, and the reason says so, e.g. `78% of zen4-128c-512G median 562,585 from run 20261014-2130 (45 nodes)`.

TOO_SLOW and SLOWER_THAN_BASELINE compare with the blessed run as usual.

A node fixed after the node stage stays out of that run's MPI stage. To include it, run its group's nodes with `--only '<hostlist>' --stage all`, or start a new full run.

## Baseline for the next maintenance

After a full run you are happy with:

```
./run_prerelease.sh --bless 20261014-2130
```

This writes `references.json` in this folder and points `results/baseline` at that run. The references are the medians of STREAM triad and copy per CPU group and of the GPU values per GPU type, taken from nodes without JOB_FAILED or WRONG_RESULT. Later runs then flag TOO_SLOW (ReFrame's own pass/fail in the node stage) and SLOWER_THAN_BASELINE. Bless a full run, not an `--only` run: the references and the baseline cover only the nodes in the run. The first run has no references, so TOO_SLOW starts after the first `--bless`; OUTLIER works from the first run.

## Long-term statistics

| Where | What |
|---|---|
| `results/<run>/metrics.csv` | every number of the run, one per row: `run_id,node,group,device,device_name,check,metric,value,unit` (STREAM, MPI_Init, barrier, each GPU by UUID, pairwise per node, the OSU curves). The baseline comparison reads this file. |
| `results/<run>/problems.csv` | every finding, including the per-node rows of SYSTEMIC issues and the MPI-stage failures |
| `results/<run>/versions.csv` | kernel, slurmd and NVIDIA driver per tested node (`node,arch,kernel,slurmd,driver`) |
| `results/reframe.db` | ReFrame's results database: every session of every run except dry runs, tagged `run=<id>` and `stage=node` or `stage=mpi` |
| `results/perflogs/sol/compute/<Test>.log` | ReFrame perflogs: `\|`-separated, one line per value, appended by every real run (dry runs write theirs into their own run directory) |

Query the database with ReFrame, from this folder:
```
rq() { RFM_ENABLE_RESULTS_STORAGE=1 RFM_SQLITE_DB_FILE=$PWD/results/reframe.db ~/rfm-4.10.4/bin/reframe "$@"; }
rq --list-stored-sessions=all                  # without =all: only the last week
rq --table-format=csv --list-stored-testcases='?run=="20261014-2130"/median:node,pvar,punit/'
rq --table-format=csv --performance-compare='?run=="20261014-2130" and stage=="node"/?run=="20260415-0910" and stage=="node"/median:node,pvar,punit/'
```

## Files

| Path | What |
|---|---|
| `site.conf` | all settings |
| `run_prerelease.sh` | the wrapper (`--help`) |
| `inventory.py` | builds `nodes.json` from `scontrol` |
| `config/sol_config.py` | ReFrame configuration |
| `tests/node_checks.py`, `tests/src/node_check.sh` | the node stage; `node_check.sh` can also be submitted by hand, see its header |
| `tests/mpi_checks.py`, `tests/src/pairbw.c` | the MPI stage (OSU is downloaded at run time) |
| `tests/src/` | also the node-stage programs (`stream_omp.c`, `mpi_hello.c`, `gpu_check.cu`) and their helper scripts |
| `report/report.py` | terminal summary, CSVs, `report.html`. It can be rerun by hand: `python3 report/report.py --run-dir results/<run> [--baseline results/<old-run>] [--parent results/<full-run>]` |
| `references.json` | written by `--bless` |
| `results/<run>/` | `nodes.json`; `stage_node.json` and `stage_mpi.json` (ReFrame reports); `reframe_node.log` and `reframe_mpi.log`; `skip_mpi.txt`; `rfm/` (job scripts and output); `report.html`, `problems.csv`, `metrics.csv`, `versions.csv` |
| `results/baseline` | link to the blessed run |
| `results/reframe.db`, `results/perflogs/` | see [Long-term statistics](#long-term-statistics) |

## Known limits

- GH200 nodes skip the GPU and MPI hello steps until `MOD_CUDA_AARCH64` and `MOD_NODE_MPI_AARCH64` are set. The skipped steps show as NOT_MEASURED.
- MI200 (sg250) and Gaudi nodes get only the device count as an accelerator check; their srun, MPI hello and STREAM steps run as on other nodes.
- sc001 and sc002 are only in `lightwork`, which allows 8 CPUs per node, so a whole-node job cannot run there and they are reported as NOT_MEASURED. To test them, raise the cap (`scontrol update PartitionName=lightwork MaxCPUsPerNode=UNLIMITED`), run `--only 'sc[001-002]'`, then set it back to 8.
- Groups with fewer than 3 nodes or GPUs are plotted but not checked for OUTLIER (except in an `--only` rerun, which borrows the parent run's median).
- Only the nodes in `MPI_GROUPS` get multi-node tests, and links are measured only inside each group.
