#!/bin/bash
# gpu_devices.sh - print one CUDA device handle per line for this node, from nvidia-smi -L:
# the MIG-... UUIDs of a GPU that has MIG instances, otherwise the GPU's own GPU-... UUID.
# Prints nothing when nvidia-smi fails or hangs (60 s). Each line is a valid CUDA_VISIBLE_DEVICES.
# nvidia-smi writes to a file, not a pipe: if it is stuck in the driver it survives the kill and
# would keep a pipe open forever.
#
# Usage: bash gpu_devices.sh

tmp=$(mktemp 2>/dev/null) || tmp=smi_L.$$
timeout -k 10 60 nvidia-smi -L > "$tmp" 2>/dev/null
awk '
    function uuid(s) {
        if (match(s, /UUID: [^)]+/))
            return substr(s, RSTART + 6, RLENGTH - 6)
        return ""
    }
    /^GPU [0-9]+:/ {
        if (gpu != "" && !mig) print gpu
        gpu = uuid($0); mig = 0; next
    }
    /^[ \t]+MIG / {
        u = uuid($0)
        if (u != "") { print u; mig = 1 }
    }
    END { if (gpu != "" && !mig) print gpu }
' "$tmp"
rm -f "$tmp"
