#!/bin/bash
# run_gpu.sh - run ./gpu_check (built by node_check.sh) on every CUDA device of this node and
# print its SOL_GPU_* lines: info, pcie and hbm one device at a time, then gemm on all devices
# at once. A device whose gpu_check failed without printing a reason gets
# SOL_GPU_ERROR reason="<test>: exit <rc>"; a failed gemm always gets reason="exit <rc>".
# gpu_check output goes to files, never a pipe: a process stuck in the driver survives the
# timeout's kill and would keep a pipe open forever.
#
# Usage: bash run_gpu.sh EXPECTED_DEVICES GEMM_SECONDS

expected=${1:-0}
secs=${2:-20}
case $secs in ''|*[!0-9]*) secs=0 ;; esac
if [ "$secs" -lt 1 ]; then
    echo "run_gpu.sh: GEMM seconds '$2' is not a positive integer, using 20" >&2
    secs=20
fi
here=$(dirname "$0")
devs=()
while read -r d; do
    if [ -n "$d" ]; then devs+=("$d"); fi
done < <(bash "$here/gpu_devices.sh")
echo "run_gpu.sh: ${#devs[@]} devices listed, Slurm gres says $expected: ${devs[*]}" >&2

# one DEV TEST LIMIT: one gpu_check test on one device; returns 124 when it timed out
one() {
    local dev=$1 test=$2 limit=$3 t0=$SECONDS out rc
    CUDA_VISIBLE_DEVICES=$dev timeout -k 10 "$limit" ./gpu_check --test "$test" > "one_$test.out"
    rc=$?
    out=$(cat "one_$test.out")
    if [ -n "$out" ]; then
        echo "$out"
    fi
    if [ "$rc" -eq 124 ] || { [ "$rc" -eq 137 ] && [ $((SECONDS - t0)) -ge "$limit" ]; }; then
        echo "SOL_TIMEOUT step=gpu seconds=$limit"
        echo "SOL_GPU_ERROR uuid=$dev reason=\"$test: timed out after $limit s\""
        return 124
    fi
    if [ "$rc" -ne 0 ] && ! grep -q '^SOL_GPU_ERROR' <<< "$out"; then
        echo "SOL_GPU_ERROR uuid=$dev reason=\"$test: exit $rc\""
    fi
    return 0
}

for dev in "${devs[@]}"; do
    for test in info pcie hbm; do
        one "$dev" "$test" 120 || break    # a hung device gets no further sequential tests
    done
done

limit=$((4 * secs + 120))
t0=$SECONDS
pids=()
for i in "${!devs[@]}"; do
    CUDA_VISIBLE_DEVICES=${devs[$i]} timeout -k 10 "$limit" ./gpu_check --test gemm --seconds "$secs" \
        > "gemm_$i.out" &
    pids[i]=$!
done
for i in "${!devs[@]}"; do
    wait "${pids[$i]}"
    rc=$?
    cat "gemm_$i.out"
    if [ "$rc" -ne 0 ] || ! grep -q '^SOL_GPU_GEMM' "gemm_$i.out"; then
        if [ "$rc" -eq 124 ] || { [ "$rc" -eq 137 ] && [ $((SECONDS - t0)) -ge "$limit" ]; }; then
            echo "SOL_TIMEOUT step=gpu seconds=$limit"
        fi
        echo "SOL_GPU_ERROR uuid=${devs[$i]} reason=\"exit $rc\""
    fi
done
