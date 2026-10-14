#!/bin/bash
# a100_inventory.sh - inventory every Slurm node carrying a feature (default a100_80):
# state, GRES, partitions, reason, and whether it sits inside a reservation
# (default oct2026_maint). Prints hostlists of usable / unusable nodes.
#
# Usage (on a login node):  ./a100_inventory.sh [reservation] [feature]

set -uo pipefail

RES=${1:-oct2026_maint}
FEAT=${2:-a100_80}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# ---- 1. nodes with the feature, one row per node ---------------------------
# sinfo -N prints one row per (node, partition); merge the partitions per node.
# Row fields: hostname|state|features|gres|reason|partition
sinfo -a -N -h -o '%n|%T|%f|%G|%E|%P' \
| awk -F'|' -v f="$FEAT" '
    $3 ~ ("(^|,)" f "(,|$)") {
        if (!($1 in st)) { order[++n] = $1; st[$1] = $2; gr[$1] = $4; re[$1] = $5; pa[$1] = $6 }
        else             { pa[$1] = pa[$1] "," $6 }
    }
    END { for (i = 1; i <= n; i++) { h = order[i]; print h "|" st[h] "|" gr[h] "|" pa[h] "|" re[h] } }
' | sort -t'|' -k1,1 > "$TMP/nodes"

if [ ! -s "$TMP/nodes" ]; then
    echo "No node carries feature '$FEAT'. Try:  sinfo -a -N -h -o '%n %f' | sort -u" >&2
    exit 1
fi

# Cross-check against scontrol, which also lists nodes that belong to no partition.
SC_COUNT=$(scontrol -o show nodes 2>/dev/null \
           | grep -cE "AvailableFeatures=([^ ]*,)?${FEAT}(,[^ ]*)?( |$)")
SI_COUNT=$(wc -l < "$TMP/nodes" | tr -d " ")
if [ -n "$SC_COUNT" ] && [ "$SC_COUNT" != "$SI_COUNT" ]; then
    echo "WARNING: scontrol sees $SC_COUNT nodes with '$FEAT' but sinfo sees $SI_COUNT." >&2
    echo "         Nodes in no partition are invisible to sinfo; check: scontrol -o show nodes | grep $FEAT" >&2
    echo >&2
fi

# ---- 2. reservation membership ---------------------------------------------
: > "$TMP/res"
if RES_LINE=$(scontrol -o show reservation "$RES" 2>/dev/null) && [ -n "$RES_LINE" ]; then
    echo "Reservation $RES:"
    printf '%s\n' "$RES_LINE" | tr ' ' '\n' \
      | grep -E '^(StartTime|EndTime|NodeCnt|PartitionName|Flags|Users|Accounts|Groups)=' \
      | sed 's/^/  /'
    echo
    RES_NODES=$(printf '%s\n' "$RES_LINE" | sed -n 's/.*[[:space:]]Nodes=\([^[:space:]]*\).*/\1/p')
    if [ -n "$RES_NODES" ] && [ "$RES_NODES" != "(null)" ]; then
        scontrol show hostnames "$RES_NODES" | sort > "$TMP/res"
    fi
else
    echo "WARNING: reservation '$RES' not found. in_res will be 'no' for every node." >&2
    echo >&2
fi

# ---- 3. classify ------------------------------------------------------------
# usable = responding, and in a state a job can run on. 'maint' means up and inside
# a MAINT reservation. Suffixes: * not responding, ~ powered down, $ in a
# maintenance reservation, @ reboot pending, # powering up, % powering down.
printf '%-10s %-14s %-22s %-24s %-6s %-6s %s\n' NODE STATE GRES PARTITIONS IN_RES USABLE REASON
awk -F'|' -v resfile="$TMP/res" -v tmp="$TMP" '
    BEGIN { while ((getline h < resfile) > 0) inres[h] = 1 }
    {
        node = $1; state = $2; gres = $3; parts = $4; reason = $5
        base = tolower(state); gsub(/[*~#%$@!^-]+$/, "", base)
        responding = (state ~ /\*/) ? 0 : 1
        usable = (responding && base ~ /^(maint|maintenance|idle|allocated|mixed|reserved|completing)$/) ? "yes" : "no"
        r = (node in inres) ? "yes" : "no"
        printf "%-10s %-14s %-22s %-24s %-6s %-6s %s\n", node, state, gres, parts, r, usable, reason
        cnt[state]++
        if (usable == "yes" && r == "yes") print node > (tmp "/good")
        if (usable == "yes" && r == "no")  print node > (tmp "/outres")
        if (usable == "no")                print node > (tmp "/bad")
    }
    END { for (s in cnt) print cnt[s], s > (tmp "/counts") }
' "$TMP/nodes"

hostlist() {  # file -> compact hostlist, or (none)
    if [ -s "$1" ]; then scontrol show hostlist "$(paste -sd, "$1")"; else echo "(none)"; fi
}
count() { if [ -s "$1" ]; then wc -l < "$1" | tr -d " "; else echo 0; fi; }

echo
echo "Nodes with '$FEAT' by state:"
sort -k2 "$TMP/counts" | awk '{ printf "  %-14s %d\n", $2, $1 }'
printf '  %-14s %d\n' TOTAL "$SI_COUNT"
echo
echo "Usable AND in $RES            : $(count "$TMP/good")"
echo "  $(hostlist "$TMP/good")"
echo "Usable but NOT in $RES        : $(count "$TMP/outres")"
echo "  $(hostlist "$TMP/outres")"
echo "Unusable (down/drain/not responding): $(count "$TMP/bad")"
echo "  $(hostlist "$TMP/bad")"
