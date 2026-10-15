#!/bin/bash
# accel_count.sh - print how many accelerators of one kind this node shows to the job.
#   nvidia  lines from gpu_devices.sh (a MIG instance counts as one device)
#   amd     KFD topology nodes with a non-zero gpu_id
#   gaudi   device indexes from hl-smi, else the /dev/accel/accel<N> nodes
#   none    0
#
# Usage: bash accel_count.sh nvidia|amd|gaudi|none

here=$(dirname "$0")
case ${1:-none} in
    nvidia)
        bash "$here/gpu_devices.sh" | grep -c .
        ;;
    amd)
        n=0
        for f in /sys/class/kfd/kfd/topology/nodes/*/gpu_id; do
            id=$(cat "$f" 2>/dev/null)
            if [ -n "$id" ] && [ "$id" != 0 ]; then
                n=$((n + 1))
            fi
        done
        echo "$n"
        ;;
    gaudi)
        n=$(timeout -k 10 60 hl-smi -Q index -f csv,noheader 2>/dev/null | grep -c '^ *[0-9]')
        if [ "$n" -eq 0 ]; then
            for f in /dev/accel/accel[0-9]*; do
                if [ -e "$f" ]; then
                    n=$((n + 1))
                fi
            done
        fi
        echo "$n"
        ;;
    *)
        echo 0
        ;;
esac
