#!/bin/bash
# h100_mix_check.sh - what a mixed A100+H100 HPL run needs from each H100 node.
# Run as root on the controller (uses ssh). Default nodes: the ones named in 256mix.sh/260mix.sh.
#
# Usage: ./h100_mix_check.sh [node ...]

NODES=${*:-scg006 scg020 scg016}

for n in $NODES; do
    echo "=================== $n"
    ssh -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR "$n" '
        echo "--- GPUs (want: 8 or 4 x H100 80GB, MIG Disabled, 0 MiB used)"
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,mig.mode.current,pci.bus_id --format=csv,noheader
        echo "--- IB devices (want: one ACTIVE HCA per GPU on an 8-way node; note NUMA of each)"
        for d in /sys/class/infiniband/*; do
            [ -e "$d" ] || { echo "NO IB DEVICES"; break; }
            h=$(basename "$d")
            printf "%-8s port1=%-10s rate=%-14s numa=%s netdev=%s\n" "$h" \
              "$(cut -d" " -f2 "$d/ports/1/state" 2>/dev/null)" \
              "$(cut -d" " -f1-2 "$d/ports/1/rate" 2>/dev/null)" \
              "$(cat "$d/device/numa_node" 2>/dev/null)" \
              "$(ls "$d/device/net" 2>/dev/null | tr "\n" ",")"
        done
        echo "--- GPU NUMA nodes (pairs with the HCA NUMA column above)"
        for b in $(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | tr "A-F" "a-f" | sed "s/^0000//"); do
            printf "%s numa=%s\n" "$b" "$(cat /sys/bus/pci/devices/$b/numa_node 2>/dev/null)"
        done
        echo "--- IPoIB interfaces that are up (the run script picks the first)"
        for i in /sys/class/net/ib*; do [ -e "$i" ] && echo "$(basename $i) $(cat $i/operstate)"; done
        echo "--- GPUDirect / IBGDA prerequisites"
        lsmod | grep -q nvidia_peermem && echo "nvidia_peermem: loaded" || echo "nvidia_peermem: NOT LOADED (GPUDirect RDMA off)"
        grep -o "PeerMappingOverride=1" /proc/driver/nvidia/params 2>/dev/null \
            && echo "PeerMappingOverride=1: set" || echo "PeerMappingOverride=1: NOT SET (IBGDA will fail; needs NVreg_RegistryDwords=PeerMappingOverride=1)"
        echo "--- CPU / memory (sets --cpus-per-task and --mem in the hetjob component)"
        echo "cores=$(nproc) mem_MiB=$(awk "/MemTotal/ {print int(\$2/1024)}" /proc/meminfo)"
        echo "--- nvidia-smi topo -m (GPU<->NIC affinity; PIX/PXB/NODE columns)"
        nvidia-smi topo -m 2>/dev/null | head -30
    ' 2>&1
    echo "--- Slurm view"
    sinfo -N -h -o '%n %T %f %G %P' -n "$n"
done
echo
echo "Reservation membership:"
scontrol show hostnames "$(scontrol -o show reservation oct2026_maint 2>/dev/null | sed -n 's/.*[[:space:]]Nodes=\([^[:space:]]*\).*/\1/p')" 2>/dev/null \
  | grep -xE "$(echo $NODES | tr ' ' '|')" | sed 's/^/  in reservation: /'
