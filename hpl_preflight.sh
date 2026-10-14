#!/bin/bash
# hpl_preflight.sh - check every node in a hostlist before an HPL run.
# Run as root on the controller (uses ssh). Prints only the nodes that fail,
# then a summary. A clean node shows: 4 GPUs, 0 MiB used, no GPU processes,
# an HCA resolved from ib0, port ACTIVE, no leftover xhpl process.
#
# Usage: ./hpl_preflight.sh ['hostlist']   default: the hpl partition's nodes
#        ./hpl_preflight.sh "$(sinfo -p hpl-256 -h -o '%N')"   for the 256-GPU mixed runs

HOSTS=${1:-$(sinfo -p hpl -h -o '%N')}

for n in $(scontrol show hostnames "$HOSTS"); do
    ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR "$n" '
        g=$(nvidia-smi -L 2>/dev/null | wc -l)
        m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
        p=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
        h=$(ls /sys/class/net/ib0/device/infiniband 2>/dev/null)
        s=$(cat /sys/class/infiniband/${h:-none}/ports/1/state 2>/dev/null | cut -d" " -f2)
        x=$(pgrep -c xhpl)
        echo "$(hostname) gpus=$g maxmem=${m:-?}MiB gpuprocs=$p hca=${h:-NONE} port=${s:-NONE} xhpl=$x"
    ' 2>&1 || echo "$n SSH_FAILED"
done | awk '
    !/ gpus=4 maxmem=0MiB gpuprocs=0 hca=mlx5_[0-9]+ port=ACTIVE xhpl=0$/ { print "BAD  " $0; bad++ }
    END { printf "%d nodes checked, %d bad\n", NR, bad + 0 }
'
