#!/bin/bash
# run_prerelease.sh - Sol pre-release check: inventory -> node stage -> MPI stage -> report.
# Settings live in site.conf; every KEY there is exported as SOL_<KEY>. Results go to
# results/<run id>/. Start it on a login node inside tmux or screen: closing the terminal stops
# ReFrame, which then cancels its jobs on SIGHUP, so the run is lost. Ctrl-C also makes ReFrame
# cancel its jobs.
#
# Usage: ./run_prerelease.sh [--stage node|mpi|all] [--only HOSTLIST] [--run-id ID] [--dry-run]
#                            [--baseline RUN_ID] [--parent RUN_ID]
#        ./run_prerelease.sh --bless RUN_ID    results/baseline -> RUN_ID, references.json from it
#   --stage mpi      continue a run whose node stage is done: needs --run-id, reuses its nodes.json
#   --only HOSTLIST  test only these nodes; runs only the node stage unless --stage is given
#   --dry-run        write the job scripts without submitting anything
#   --baseline ID    compare with results/ID (default: results/baseline when it exists)
#   --parent ID      with --only: compare with results/ID where this run has too few nodes of a group
#                    (default: the newest full run in results/ that finished its node stage)
# REFRAME and PYTHON set in the environment override site.conf, e.g.
#   REFRAME=~/rfm-4.10.4/bin/reframe ./run_prerelease.sh
# Offline-test settings in the environment (SOL_SCONTROL_*_FILE, SOL_MODULES_SYSTEM) are ignored
# unless SOL_OFFLINE=1.
# Exit: report.py's code (0 no problems; 1 a node problem from UNAVAILABLE to TOO_SLOW or a failed
#       MPI-stage test), 2 setup error, 3 a stage did not run or ReFrame stopped early, 130 Ctrl-C.

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }
die() { echo "run_prerelease.sh: $*" >&2; exit 2; }
need() { [ $# -ge 2 ] && [ -n "$2" ] && [ "${2#-}" = "$2" ] || die "$1 needs a value"; }

# lowercase names for the script's own variables: sourcing site.conf only sets uppercase keys
stage=all stage_set=0 only='' run_id='' dry=0 baseline='' bless='' parent=''
while [ $# -gt 0 ]; do
    case "$1" in
        --stage)      need "$@"; stage=$2; stage_set=1; shift 2 ;;
        --stage=*)    stage=${1#*=}; stage_set=1; shift ;;
        --only)       need "$@"; only=$2; shift 2 ;;
        --only=*)     only=${1#*=}; shift ;;
        --run-id)     need "$@"; run_id=$2; shift 2 ;;
        --run-id=*)   run_id=${1#*=}; shift ;;
        --baseline)   need "$@"; baseline=$2; shift 2 ;;
        --baseline=*) baseline=${1#*=}; shift ;;
        --parent)     need "$@"; parent=$2; shift 2 ;;
        --parent=*)   parent=${1#*=}; shift ;;
        --bless)      need "$@"; bless=$2; shift 2 ;;
        --bless=*)    bless=${1#*=}; shift ;;
        --dry-run)    dry=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        *)            die "unknown argument '$1' (see --help)" ;;
    esac
done
case "$stage" in node|mpi|all) ;; *) die "--stage must be node, mpi or all" ;; esac
for id in "$run_id" "$baseline" "$bless" "$parent"; do
    case "$id" in
        .|..|*[!A-Za-z0-9._-]*) die "run ids may only use letters, digits, '.', '_' and '-' (got '$id')" ;;
    esac
done
if [ -n "$bless" ] && { [ "$stage_set" = 1 ] || [ -n "$only$run_id$baseline$parent" ] || [ "$dry" = 1 ]; }; then
    die "--bless takes no other options"
fi
[ "$stage" = mpi ] && [ -z "$run_id" ] && die "--stage mpi continues an earlier run: give its --run-id"
[ "$stage" = mpi ] && [ -n "$only" ] &&
    die "--only selects the nodes of a new run; --stage mpi reuses an existing nodes.json (use --stage all)"
[ -n "$parent" ] && [ -z "$only" ] && [ "$stage" != mpi ] &&
    die "--parent goes with --only: it names the full run a rerun is compared with"

# --- set up: suite dir, site.conf -> SOL_*, tools
abspath() { case "$1" in */*) case "$1" in /*) echo "$1" ;; *) echo "$PWD/$1" ;; esac ;; *) echo "$1" ;; esac; }
env_reframe=${REFRAME:+$(abspath "$REFRAME")}
env_python=${PYTHON:+$(abspath "$PYTHON")}
src=${BASH_SOURCE[0]}
while [ -L "$src" ]; do
    dir=$(cd -P "$(dirname "$src")" && pwd)
    src=$(readlink "$src")
    case "$src" in /*) ;; *) src=$dir/$src ;; esac
done
cd -P "$(dirname "$src")" || die "cannot cd to the suite directory"
[ -f site.conf ] || die "site.conf not found in $PWD"

# shellcheck source=site.conf
source ./site.conf || die "site.conf has an error"
REFRAME=${env_reframe:-${REFRAME:-reframe}}
PYTHON=${env_python:-${PYTHON:-python3}}
for key in $(grep -oE '^[A-Za-z_][A-Za-z0-9_]*=' site.conf | tr -d '='); do
    export "SOL_$key=${!key}"
done
# durations in ReFrame's grammar (<d>d<h>h<m>m<s>s, whole numbers, at least one part; a bare number is
# seconds): a typo would otherwise make ReFrame skip a test file, for the MPI keys only after the node stage
duration_re='^([0-9]+d)?([0-9]+h)?([0-9]+m)?([0-9]+s)?$' seconds_re='^[0-9]+$'
for key in NODE_TIME_LIMIT MPI_TIME_LIMIT MAX_PENDING MPI_MAX_PENDING; do
    v=SOL_$key
    val=${!v}
    [ -z "$val" ] && continue                      # empty: the tests use their own default
    if [[ $val =~ $seconds_re ]]; then
        echo "WARNING: $key=\"$val\" in site.conf has no unit, so ReFrame reads it as $val seconds;" \
             "add m or h if you meant minutes or hours" >&2
    elif ! [[ $val =~ $duration_re ]]; then
        die "$key=\"$val\" in site.conf is not a ReFrame duration: write whole numbers with d, h, m or s," \
            "e.g. 30m, 4h or 1h30m"
    fi
done
unset SOL_SKIP_NODES_FILE SOL_REFERENCES
# captured scontrol output and no Lmod are for offline tests; a real run must not inherit them
if [ "${SOL_OFFLINE:-0}" != 1 ]; then
    for v in SOL_SCONTROL_NODES_FILE SOL_SCONTROL_RES_FILE SOL_SCONTROL_PARTS_FILE SOL_MODULES_SYSTEM; do
        if [ -n "${!v+x}" ]; then
            echo "WARNING: ignoring $v=${!v}, an offline-test setting (SOL_OFFLINE=1 keeps it)" >&2
            unset "$v"
        fi
    done
fi
results=$PWD/results
export SOL_DB_FILE=$results/reframe.db
[ -f references.json ] && export SOL_REFERENCES=$PWD/references.json

"$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 8))' 2>/dev/null ||
    die "PYTHON='$PYTHON' does not run or is older than 3.8 (set PYTHON in site.conf)"

if [ -n "$bless" ]; then
    run_dir=$results/$bless
    [ -f "$run_dir/nodes.json" ] || die "results/$bless has no nodes.json"
    [ -f "$run_dir/stage_node.json" ] || [ -f "$run_dir/stage_mpi.json" ] ||
        die "results/$bless has no stage_node.json or stage_mpi.json: not a completed run"
    # an --only run replaces the cluster-wide references and baseline with its few nodes: say so
    sel=$("$PYTHON" -c 'import json, sys
nodes = json.load(open(sys.argv[1])).get("nodes", [])
if any(n.get("selected") is False for n in nodes):
    print(sum(1 for n in nodes if n.get("testable")))' "$run_dir/nodes.json" 2>/dev/null)
    [ -e "$results/baseline" ] && [ ! -L "$results/baseline" ] &&
        die "results/baseline exists and is not a symlink; move it away first"
    rm -f references.json.new
    "$PYTHON" report/report.py --run-dir "$run_dir" --write-references references.json.new --quiet
    rc=$?
    # report.py exits 1 when the run has no reference values (it still writes the file)
    if [ "$rc" != 0 ] || ! "$PYTHON" -c 'import json, sys; json.load(open(sys.argv[1]))' \
            references.json.new 2>/dev/null; then
        rm -f references.json.new
        die "report.py found no reference values in results/$bless (exit $rc); nothing changed"
    fi
    mv -f references.json.new references.json || die "cannot write references.json"
    ln -sfn "$bless" "$results/baseline" || die "cannot link results/baseline"
    echo "results/baseline -> $bless; references.json written from it"
    [ -n "$sel" ] && echo "WARNING: results/$bless is an --only run: references.json and results/baseline" \
        "now cover only its $sel tested node(s); bless a full run to restore cluster-wide references" >&2
    exit 0
fi

if ! rfm_version=$("$REFRAME" --version 2>/dev/null); then
    cat >&2 <<EOF
run_prerelease.sh: '$REFRAME --version' failed: ReFrame 4.10.4 is not installed or not found.
Install it once with Python 3.10 or newer (see README.md), for example:
    python3.12 -m venv ~/rfm-4.10.4 && ~/rfm-4.10.4/bin/pip install reframe-hpc==4.10.4
then set REFRAME="\$HOME/rfm-4.10.4/bin/reframe" in site.conf.
EOF
    exit 2
fi
case "$rfm_version" in 4.10.*) ;; *) echo "WARNING: written for ReFrame 4.10.4, found '$rfm_version'" >&2 ;; esac
[ -n "${LOADEDMODULES:-}" ] &&
    echo "WARNING: modules loaded in this shell: ${LOADEDMODULES//:/ } (informational: every job starts with 'module purge')" >&2

run_node=1 run_mpi=1
[ "$stage" = mpi ] && run_node=0
[ "$stage" = node ] && run_mpi=0
[ -n "$only" ] && [ "$stage_set" = 0 ] && run_mpi=0

if [ -z "$run_id" ]; then
    run_id=$(date +%Y%m%d-%H%M)
    [ -e "$results/$run_id" ] && run_id=$(date +%Y%m%d-%H%M%S)
fi
run_dir=$results/$run_id
if [ "$run_node" = 1 ]; then
    [ -e "$run_dir" ] && die "results/$run_id already exists; pick another --run-id (or --stage mpi to continue it)"
elif [ ! -f "$run_dir/nodes.json" ]; then
    die "results/$run_id/nodes.json not found: --stage mpi needs a run whose node stage is done"
elif [ "$dry" = 1 ] && [ -f "$run_dir/stage_mpi.json" ]; then
    die "results/$run_id already ran its MPI stage; a dry run would wipe its stage directories"
fi
[ -n "$baseline" ] && [ ! -d "$results/$baseline" ] && die "--baseline: results/$baseline not found"
if [ -n "$parent" ]; then
    [ "$parent" = "$run_id" ] && die "--parent names this run; give the full run to compare it with"
    [ -d "$results/$parent" ] || die "--parent: results/$parent not found"
fi
# the --only hostlist a run was made with ("" = a full run)
run_only() { "$PYTHON" -c 'import json, sys; print(json.load(open(sys.argv[1])).get("only") or "")' "$1" 2>/dev/null; }
[ -n "$parent" ] && [ "$run_node" = 0 ] && [ -z "$(run_only "$run_dir/nodes.json")" ] &&
    die "--parent goes with an --only run, and results/$run_id is a full run"
mkdir -p "$run_dir" || die "cannot create $run_dir"
export SOL_INVENTORY=$run_dir/nodes.json

# Ctrl-C: ReFrame cancels its jobs before it exits, but a second Ctrl-C interrupts that
# shellcheck disable=SC2329  # called by the INT trap
on_interrupt() {
    echo "run_prerelease.sh: interrupted; results so far are in $run_dir" >&2
    if [ "$dry" = 0 ]; then
        echo "  ReFrame cancels its jobs on Ctrl-C unless a second Ctrl-C stopped it; list any left over:" >&2
        echo "    squeue -h -u $(id -un) -o '%i %j %T' | awk '\$2 ~ /^rfm_Sol/'" >&2
    fi
    exit 130
}
trap on_interrupt INT

echo "== Sol pre-release check, run $run_id, ReFrame $rfm_version$([ "$dry" = 1 ] && echo ', dry run')"
if [ "$run_node" = 1 ]; then
    inv_args=(--out "$SOL_INVENTORY")
    [ -n "$only" ] && inv_args+=(--only "$only")
    "$PYTHON" inventory.py "${inv_args[@]}"
    rc=$?
    [ "$rc" = 0 ] || die "inventory.py failed (exit $rc); nothing was submitted"
else
    echo "== reusing results/$run_id/nodes.json"
fi

# An --only rerun of a few nodes has too few of them to compare with each other: report.py --parent takes
# the medians and the NVIDIA driver majority from a full run where a group has fewer than 3 nodes or GPUs.
# Default: the newest run in results/ whose nodes.json says it was a full run and whose node stage finished.
newest_full_run() {
    "$PYTHON" - "$results" "$run_id" 2>/dev/null <<'EOF'
import json, os, sys
results, this = sys.argv[1], sys.argv[2]
best = None
for rid in os.listdir(results):
    d = os.path.join(results, rid)
    if rid == this or os.path.islink(d) or not os.path.isfile(os.path.join(d, 'stage_node.json')):
        continue
    try:
        with open(os.path.join(d, 'nodes.json')) as fp:
            inv = json.load(fp)
    except (OSError, ValueError):
        continue
    if isinstance(inv, dict) and inv.get('only') == '':
        key = (str(inv.get('generated') or ''), os.path.getmtime(os.path.join(d, 'nodes.json')), rid)
        best = max(best, (key, rid)) if best else (key, rid)
if best:
    print(best[1])
EOF
}
parent_dir=
if [ "$dry" = 0 ] && [ -n "$(run_only "$SOL_INVENTORY")" ]; then
    what="a group with fewer than 3 nodes or GPUs here is compared with that run"
    if [ -n "$parent" ]; then
        parent_dir=$results/$parent
        echo "== parent run $parent (--parent): $what"
    elif pick=$(newest_full_run) && [ -n "$pick" ]; then
        parent_dir=$results/$pick
        echo "== parent run $pick (newest full run in results/; --parent ID picks another): $what"
    else
        echo "== no parent run: results/ has no full run with a node stage, so groups with fewer than 3 nodes or" \
             "GPUs here get no OUTLIER or NVIDIA driver check"
    fi
fi

notes=()
stage_failed=0
rfm_failed=0
# rfm_stage STAGE TESTFILE: one ReFrame session; returns 1 when it wrote no report or stopped early.
# Only the MPI stage lets jobs wait in ReqNodeNotAvail: its jobs queue behind each other on the
# same nodes, so Slurm gives the waiting ones that reason.
rfm_stage() {
    local name=$1 tests=$2 action=-r report=$run_dir/stage_$1.json log=$run_dir/reframe_$1.log rc
    local ignore_rnna=0 perflogs=$results/perflogs
    [ "$name" = mpi ] && ignore_rnna=1
    if [ "$dry" = 1 ]; then
        # a dry run measures nothing: its perflog lines (empty values) stay in the run dir
        action=--dry-run
        report=$run_dir/dryrun_$name.json
        perflogs=$run_dir/perflogs
    fi
    rm -f "$report"
    echo "== $name stage: $tests"
    RFM_IGNORE_REQNODENOTAVAIL=$ignore_rnna "$REFRAME" -C config/sol_config.py --system sol -c "$tests" "$action" \
        --prefix "$run_dir/rfm" --report-file "$report" --perflogdir "$perflogs" \
        --session-extras "run=$run_id,stage=$name" 2>&1 | tee -i "$log"
    rc=${PIPESTATUS[0]}
    [ "$rc" = 0 ] || rfm_failed=1
    if grep -q 'skipping test file' "$log"; then
        notes+=("ERROR: $name stage: ReFrame skipped a test file that failed to load; see $log")
        stage_failed=1
    fi
    # killed, or the session ended on an error such as too many sacct failures: jobs may be left over
    if [ "$rc" -ge 128 ] || grep -q 'run session stopped' "$log"; then
        notes+=("ERROR: $name stage: ReFrame stopped early (exit $rc; see $log)")
        [ "$dry" = 1 ] || notes+=(
            "  its Slurm jobs may still be queued or running; list them, and cancel them before a rerun:"
            "    squeue -h -u $(id -un) -o '%i %j %T' | awk '\$2 ~ /^rfm_Sol/'"
            "    squeue -h -u $(id -un) -o '%i %j' | awk '\$2 ~ /^rfm_Sol/ {print \$1}' | xargs -r scancel")
        stage_failed=1
        return 1
    fi
    if [ -f "$report" ]; then
        notes+=("$name stage: ReFrame exit $rc; report $report")
        return 0
    fi
    notes+=("$name stage: ReFrame exit $rc and wrote no report; see $log")
    return 1
}

if [ "$run_node" = 1 ] && ! rfm_stage node tests/node_checks.py; then
    notes+=("ERROR: the node stage did not run to the end$([ "$run_mpi" = 1 ] && echo ', so the MPI stage was not started')")
    stage_failed=1 run_mpi=0
fi

if [ "$run_mpi" = 1 ] && [ "$dry" = 0 ]; then
    [ -f "$run_dir/stage_node.json" ] ||
        notes+=("WARNING: results/$run_id has no stage_node.json: the MPI stage ran without a node-stage skip list")
    rm -f "$run_dir/skip_mpi.txt"
    "$PYTHON" report/report.py --run-dir "$run_dir" --stage node \
        --failed-nodes-out "$run_dir/skip_mpi.txt" --quiet
    if [ -f "$run_dir/skip_mpi.txt" ]; then
        export SOL_SKIP_NODES_FILE=$run_dir/skip_mpi.txt
        echo "== $(grep -c . "$SOL_SKIP_NODES_FILE") node(s) failed the node stage and stay out of the MPI stage"
    else
        notes+=("ERROR: report.py wrote no skip_mpi.txt, so the MPI stage was not started; rerun it with --stage mpi --run-id $run_id")
        stage_failed=1 run_mpi=0
    fi
fi
[ "$run_mpi" = 1 ] && rfm_stage mpi tests/mpi_checks.py

echo
if [ "$dry" = 1 ]; then
    n=$(find "$run_dir/rfm/stage" -name rfm_job.sh 2>/dev/null | wc -l | tr -d ' ')
    echo "Dry run $run_id: $n job scripts in $run_dir/rfm/stage/sol/compute/<environ>/<test>/rfm_job.sh"
    printf '  %s\n' "${notes[@]}"
    [ "$stage_failed" = 0 ] && [ "$rfm_failed" = 0 ] && exit 0
    exit 1
fi

report_args=(--run-dir "$run_dir")
if [ -n "$baseline" ]; then
    [ "$results/$baseline" -ef "$run_dir" ] || report_args+=(--baseline "$results/$baseline")
elif [ -e "$results/baseline" ] && ! [ "$results/baseline" -ef "$run_dir" ]; then
    report_args+=(--baseline "$results/baseline")
fi
[ -n "$parent_dir" ] && report_args+=(--parent "$parent_dir")
"$PYTHON" report/report.py "${report_args[@]}"
rc=$?
echo
echo "Run $run_id: $run_dir"
printf '  %s\n' "${notes[@]}"
[ "$stage_failed" = 1 ] && exit 3
exit "$rc"
