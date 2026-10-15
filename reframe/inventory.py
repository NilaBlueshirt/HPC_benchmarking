#!/usr/bin/env python3
# inventory.py - write nodes.json, the node list every Sol pre-release test is built from.
# Reads 'scontrol -a -o show node', 'scontrol -a -o show partition' and
# 'scontrol -o show reservation NAME', and decides for each node whether it is testable: not in
# EXCLUDE_NODES, a usable Slurm state, in the reservation, in --only when that is given, and in a
# submit partition whose MaxCPUsPerNode and MaxTime allow a whole-node job of NODE_TIME_LIMIT.
# Every node record also keeps the kernel and slurmd version Slurm reports (OS=, Version=), which report.py
# uses as the cluster-wide majority for MISMATCH; "only" records the --only hostlist ("" for a full run).
# Prints a summary to stderr.
# Settings default to the SOL_* variables that run_prerelease.sh exports from site.conf.
#
# Usage: python3 inventory.py --out results/<run>/nodes.json [--only 'scc032,scg016']
#            [--reservation NAME] [--exclude HOSTLIST] [--partitions "general public ..."]
#        Offline: --scontrol-nodes-file F --scontrol-res-file F --scontrol-parts-file F (captured
#        'scontrol -o' output), or the same through SOL_SCONTROL_NODES_FILE / SOL_SCONTROL_RES_FILE /
#        SOL_SCONTROL_PARTS_FILE.
# Exit: 0 ok, 1 no testable node, 2 scontrol failed (or its captured file is unreadable) or bad input.

import argparse
import datetime
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tests'))
from sol_common import compress_hostlist, env, expand_hostlist  # noqa: E402

# site.conf values, used only when SOL_<KEY> is not set at all (set but empty means "none")
DEFAULTS = {'RESERVATION': 'oct2026_maint',
            'EXCLUDE_NODES': 'sg[001-050],sdg051,sg[235-239],scg[009-012]',
            'SUBMIT_PARTITIONS': 'general public htc highmem lightwork fpga gaudi arm'}

# a value runs until whitespace followed by the next "Key=" (Reason= and OS= contain spaces)
KV_RE = re.compile(r'(\w+)=(.*?)(?=\s+[A-Za-z][\w/]*=|$)')
BAD_STATES = ('DOWN', 'DRAIN', 'FAIL', 'NOT_RESPONDING', 'POWER', 'FUTURE', 'INVAL', 'UNKNOWN',
              'REBOOT')
CPU_MODELS = ('zen3', 'zen4', 'zen5', 'sapphirerapids', 'emeraldrapids', 'icelake', 'cascadelake')
GPU_MODELS = ('a100_80', 'a100_40', 'a100_20', 'a30', 'h100_80', 'h100nvl', 'h200', 'l40s',
              'gh200', 'mi200', 'hl225')
SCONTROL_TIMEOUT = 120


class InventoryError(Exception):
    pass


def setting(key):
    name = 'SOL_' + key
    return os.environ[name].strip() if name in os.environ else DEFAULTS[key]


def parse_kv(line):
    '''Key=value pairs of one 'scontrol -o' line. The first copy of a key wins: free text such as
    Reason= or Comment= comes after the real fields and may itself contain "State=...".'''
    kv = {}
    for m in KV_RE.finditer(line.strip()):
        kv.setdefault(m.group(1), m.group(2).strip())
    return kv


def num(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def listval(value):
    return [] if not value or value == '(null)' else [x for x in value.split(',') if x]


def os_kernel(os_field):
    '''The kernel release from OS= ("Linux 5.14.0-503.14.1.el9_5.x86_64 #1 SMP ..." -> its second token),
    the same text as 'uname -r' on the node; "" when Slurm reports no OS (a node that never registered).'''
    tokens = (os_field or '').split()
    return tokens[1] if len(tokens) > 1 else ''


def slurmd_version(version):
    '''Version= as Slurm reports it, the same text as 'slurmd -V' prints after "slurm"; "" when missing.'''
    version = (version or '').strip()
    return '' if version in ('(null)', 'N/A') else version


def hostset(expr, what):
    try:
        return set(expand_hostlist(expr))
    except ValueError:
        raise InventoryError('cannot parse %s hostlist %r' % (what, expr))


def scontrol_text(cmd, path):
    '''Output of the scontrol command, or the captured text in path when given.'''
    if path:
        try:
            with open(path) as fp:
                return fp.read()
        except (IOError, OSError) as e:
            raise InventoryError('cannot read %s: %s' % (path, e))
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, timeout=SCONTROL_TIMEOUT)
    except OSError as e:
        raise InventoryError('cannot run %s: %s' % (' '.join(cmd), e))
    except subprocess.TimeoutExpired:
        raise InventoryError('%s did not answer within %d s' % (' '.join(cmd), SCONTROL_TIMEOUT))
    if p.returncode != 0:
        raise InventoryError('%s failed (exit %d): %s' % (' '.join(cmd), p.returncode,
                                                          (p.stderr or p.stdout).strip()))
    return p.stdout


def split_gres(text):
    '''Split a Gres= value on commas outside (...): "gpu:a:2(S:0,1),gpu:b:1" is two entries.'''
    parts, depth, buf = [], 0, ''
    for ch in text:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth = max(0, depth - 1)
        if ch == ',' and depth == 0:
            parts.append(buf)
            buf = ''
        else:
            buf += ch
    parts.append(buf)
    return [p.strip() for p in parts if p.strip()]


def gpu_entries(gres_raw):
    '''[(type, count), ...] for the gpu entries of Gres=, socket/index suffixes dropped.'''
    out = []
    for entry in split_gres(gres_raw):
        fields = [f for f in re.sub(r'\(.*?\)', '', entry).split(':') if f != 'no_consume']
        if fields[0] != 'gpu':
            continue
        count = 1
        if len(fields) > 1 and fields[-1].isdigit():
            count = int(fields.pop())
        if count > 0:
            out.append((':'.join(fields[1:]), count))
    return out


def accel_of(entries, features):
    gtype = entries[0][0].lower() if entries else ''
    if gtype.startswith('hl') or 'gaudi' in features:
        return 'gaudi'
    if not entries:
        return 'none'
    return 'amd' if gtype.startswith('mi') else 'nvidia'


def gpu_model_of(entries, features):
    if not entries:
        return ''
    gtype = entries[0][0].lower()
    base = gtype.split('.')[0]                     # a100.40gb -> a100, h200.35gb -> h200
    for model in GPU_MODELS:                       # a feature refines the gres type: l40 + l40s
        if model in features and model.startswith(base) and \
                (model == base or not model[len(base)].isdigit()):
            return model
    if base in GPU_MODELS:
        return base
    sized = re.match(r'^(.+)\.(\d+)gb$', gtype)
    if sized and '%s_%s' % sized.groups() in GPU_MODELS:
        return '%s_%s' % sized.groups()
    return gtype or 'gpu'


def cpu_model_of(arch, features):
    for feat in features:
        if feat in CPU_MODELS:
            return feat
    return 'grace' if arch == 'aarch64' else 'unknown'


def mem_gb_class(real_memory_mb):
    '''RealMemory (MiB) rounded up to whole 64 GiB: the installed size (515516 -> 512).'''
    return -(-real_memory_mb // 65536) * 64


def duration_minutes(text):
    '''A ReFrame duration ("20m", "1h30m", "90" = seconds) in minutes, rounded up like Slurm
    rounds --time; None when ReFrame would reject it.'''
    try:
        secs = float(text)
    except ValueError:
        m = re.match(r'^(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$', text)
        if not m:
            return None
        d, h, mi, s = [int(x or 0) for x in m.groups()]
        secs = ((d * 24 + h) * 60 + mi) * 60 + s
    if not 0 <= secs < float('inf'):
        return None
    return int(-(-secs // 60))


def slurm_minutes(text):
    '''A time limit as scontrol prints it ("1-00:00:00", "04:00:00") in minutes; None for UNLIMITED.'''
    m = re.match(r'^(?:(\d+)-)?(\d+):(\d+):(\d+)$', text or '')
    if not m:
        return None
    d, h, mi, s = [int(x or 0) for x in m.groups()]
    return (d * 24 + h) * 60 + mi + (1 if s else 0)


def partition_misfit(part, cpus, ctx):
    '''Why a whole-node job on a node with this many CPUs cannot run in partition part ([] if it can).
    A partition that 'scontrol show partition' does not list counts as fitting.'''
    info = ctx['parts'].get(part, {})
    why = []
    max_cpus = num(info.get('MaxCPUsPerNode'))                    # UNLIMITED -> 0
    if 0 < max_cpus < cpus:
        why.append('partition %s allows %d CPUs per node (MaxCPUsPerNode), a whole-node job cannot '
                   'run there' % (part, max_cpus))
    max_time = slurm_minutes(info.get('MaxTime'))
    if max_time is not None and ctx['time_limit_min'] is not None and max_time < ctx['time_limit_min']:
        why.append('partition %s allows jobs of at most %s (MaxTime), less than NODE_TIME_LIMIT=%s'
                   % (part, info['MaxTime'], ctx['time_limit']))
    return why


def node_record(kv, ctx):
    name = kv['NodeName']
    features = (listval(kv.get('AvailableFeatures')) or listval(kv.get('ActiveFeatures'))
                or listval(kv.get('Features')))
    feats = [f.lower() for f in features]
    arch = kv.get('Arch', '')
    arch = '' if arch == '(null)' else arch
    gres_raw = kv.get('Gres', '')
    gres_raw = '' if gres_raw == '(null)' else gres_raw
    entries = gpu_entries(gres_raw)
    sockets, cps, tpc = num(kv.get('Sockets')), num(kv.get('CoresPerSocket')), \
        num(kv.get('ThreadsPerCore'))
    cores = sockets * cps
    cpus = num(kv.get('CPUTot'))
    mem = num(kv.get('RealMemory'))
    cpu_model = cpu_model_of(arch, feats)
    mem_class = mem_gb_class(mem)
    partitions = listval(kv.get('Partitions'))
    state = kv.get('State') or 'UNKNOWN'
    reason = kv.get('Reason', '')

    why = []
    bad = [b for b in BAD_STATES if b in state.upper()] or (['*'] if '*' in state else [])
    if bad:
        why.append('Slurm state %s%s' % (state, ': ' + reason if reason else ''))
    in_res = ctx['res_nodes'] is None or name in ctx['res_nodes']
    if not in_res:
        why.append('not in reservation %s' % ctx['reservation'])
    if not partitions:
        why.append('in no Slurm partition')
    if cores <= 0:
        why.append('Slurm reports no cores')
    excluded = name in ctx['exclude']
    available = not why
    selected = ctx['only'] is None or name in ctx['only']

    # the first SUBMIT_PARTITIONS entry with the node whose limits allow the whole-node job
    candidates = [p for p in ctx['submit'] if p in partitions] or partitions
    misfit = dict((p, partition_misfit(p, cpus, ctx)) for p in candidates)
    fits = [p for p in candidates if not misfit[p]]
    untestable = ''
    if not excluded and available and selected and not fits:
        untestable = '; '.join(w for p in candidates for w in misfit[p])
    return {
        'name': name,
        'arch': arch,
        'cpus': cpus,
        'sockets': sockets,
        'cores_per_socket': cps,
        'threads_per_core': tpc,
        'cores': cores,
        'real_memory_mb': mem,
        'partitions': partitions,
        'state': state,
        'reason': reason,
        'os_kernel': os_kernel(kv.get('OS')),
        'slurmd_version': slurmd_version(kv.get('Version')),
        'features': features,
        'gres_raw': gres_raw,
        'gres_request': ','.join('gpu:%s:%d' % (t, c) if t else 'gpu:%d' % c for t, c in entries),
        'gpu_count': sum(c for _, c in entries),
        'accel': accel_of(entries, feats),
        'gpu_model': gpu_model_of(entries, feats),
        'cpu_model': cpu_model,
        'ib': 'ib' in feats,
        'mem_gb_class': mem_class,
        'cpu_group': '%s-%dc-%dG' % (cpu_model, cores, mem_class),
        'submit_partition': fits[0] if fits else (candidates[0] if candidates else ''),
        'in_reservation': in_res,
        'excluded': excluded,
        'exclude_reason': ctx['exclude_reason'] if excluded else '',
        'available': available,
        'unavailable_reason': '; '.join(why),
        'selected': selected,
        'untestable_reason': untestable,
        'testable': not excluded and available and selected and not untestable,
    }


def natural_key(name):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', name)]


def read_reservation(name, path):
    text = scontrol_text(['scontrol', '-o', 'show', 'reservation', name], path)
    for line in text.splitlines():
        kv = parse_kv(line)
        if kv.get('ReservationName') == name:
            return kv
    raise InventoryError('reservation %s not found in %s' % (name, path or 'scontrol output'))


def read_partitions(path):
    '''{PartitionName: its key=value dict} from 'scontrol -a -o show partition'.'''
    text = scontrol_text(['scontrol', '-a', '-o', 'show', 'partition'], path)
    parts = {}
    for line in text.splitlines():
        kv = parse_kv(line)
        if kv.get('PartitionName'):
            parts[kv['PartitionName']] = kv
    return parts


def reservation_warnings(res):
    out = []
    if res.get('State', 'ACTIVE') != 'ACTIVE':
        out.append('reservation %s is %s (starts %s): jobs wait until it starts'
                   % (res['ReservationName'], res.get('State'), res.get('StartTime', '?')))
    try:
        end = datetime.datetime.strptime(res.get('EndTime', ''), '%Y-%m-%dT%H:%M:%S')
        if end - datetime.datetime.now() < datetime.timedelta(hours=3):
            out.append('reservation %s ends %s: jobs that cannot finish before then will not start'
                       % (res['ReservationName'], res['EndTime']))
    except ValueError:
        pass
    return out


def summary(nodes, ctx, out_path, source, res, warnings):
    excl = [n for n in nodes if n['excluded']]
    unsel = [n for n in nodes if not n['excluded'] and not n['selected']]
    unav = [n for n in nodes if not n['excluded'] and n['selected'] and not n['available']]
    untest = [n for n in nodes if n['untestable_reason']]
    test = [n for n in nodes if n['testable']]
    head = 'inventory: %d nodes from %s' % (len(nodes), source)
    if res:
        head += ', reservation %s (%s, ends %s)' % (
            res['ReservationName'], res.get('State', '?'), res.get('EndTime', '?'))
    lines = [head, '  testable %d | excluded %d | unavailable %d | untestable %d'
             % (len(test), len(excl), len(unav), len(untest))]
    if ctx['only'] is not None:
        lines[-1] += ' | not in --only %d' % len(unsel)
    if test:
        by = {}
        for n in test:
            by[n['accel']] = by.get(n['accel'], 0) + 1
        lines.append('  testable by accelerator: ' +
                     ', '.join('%s %d' % kv for kv in sorted(by.items())))
    for title, group, key in (('unavailable', unav, 'unavailable_reason'),
                              ('untestable', untest, 'untestable_reason')):
        if group:
            lines.append('  %s (not tested):' % title)
            width = max(len(n['name']) for n in group)
            lines += ['    %-*s  %s' % (width, n['name'], n[key]) for n in group]
    excl_bad = [n['name'] for n in excl if not n['available']]
    if excl_bad:
        lines.append('  note: excluded nodes that are unavailable now: ' + compress_hostlist(excl_bad))
    lines += ['  WARNING: ' + w for w in warnings]
    lines.append('  wrote ' + out_path)
    sys.stderr.write('\n'.join(lines) + '\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description='Write nodes.json for the Sol pre-release tests.')
    ap.add_argument('--out', required=True, help='nodes.json to write')
    ap.add_argument('--reservation', help='Slurm reservation (default SOL_RESERVATION; "" = none)')
    ap.add_argument('--exclude', help='hostlist not to test (default SOL_EXCLUDE_NODES)')
    ap.add_argument('--only', help='hostlist: test only these nodes')
    ap.add_argument('--partitions', help='submit partitions, preferred first '
                                         '(default SOL_SUBMIT_PARTITIONS)')
    ap.add_argument('--scontrol-nodes-file', help="captured 'scontrol -o show node' output")
    ap.add_argument('--scontrol-res-file', help="captured 'scontrol -o show reservation' output")
    ap.add_argument('--scontrol-parts-file', help="captured 'scontrol -o show partition' output")
    args = ap.parse_args(argv)

    nodes_file = args.scontrol_nodes_file or os.environ.get('SOL_SCONTROL_NODES_FILE', '').strip()
    res_file = args.scontrol_res_file or os.environ.get('SOL_SCONTROL_RES_FILE', '').strip()
    parts_file = args.scontrol_parts_file or os.environ.get('SOL_SCONTROL_PARTS_FILE', '').strip()
    reservation = (setting('RESERVATION') if args.reservation is None else args.reservation).strip()
    if args.exclude is None:
        exclude_expr, exclude_reason = setting('EXCLUDE_NODES'), 'validated by HPL (EXCLUDE_NODES)'
    else:
        exclude_expr, exclude_reason = args.exclude, 'excluded with --exclude'
    submit = (setting('SUBMIT_PARTITIONS') if args.partitions is None else args.partitions).split()
    time_limit = env('NODE_TIME_LIMIT', '20m').strip()            # what node_checks.py uses
    time_limit_min = duration_minutes(time_limit)
    warnings = []
    if time_limit_min is None:
        warnings.append('NODE_TIME_LIMIT=%s is not a ReFrame duration: partition MaxTime not checked'
                        % time_limit)
    try:
        exclude = hostset(exclude_expr, 'exclude')
        only = hostset(args.only, '--only') if args.only and args.only.strip() else None
        text = scontrol_text(['scontrol', '-a', '-o', 'show', 'node'], nodes_file)
        parts = read_partitions(parts_file)
        res, res_nodes = None, None
        if reservation:
            res = read_reservation(reservation, res_file)
            spec = res.get('Nodes', '')
            if spec.upper() != 'ALL':
                res_nodes = hostset('' if spec == '(null)' else spec, 'reservation')
            warnings += reservation_warnings(res)
    except InventoryError as e:
        sys.stderr.write('inventory.py: ERROR: %s\n' % e)
        return 2

    ctx = {'reservation': reservation, 'res_nodes': res_nodes, 'exclude': exclude,
           'exclude_reason': exclude_reason, 'only': only, 'submit': submit, 'parts': parts,
           'time_limit': time_limit, 'time_limit_min': time_limit_min}
    records = {}
    for line in text.splitlines():
        kv = parse_kv(line)
        if kv.get('NodeName'):
            records[kv['NodeName']] = node_record(kv, ctx)
    nodes = [records[k] for k in sorted(records, key=natural_key)]

    unlisted = sorted(set(n['submit_partition'] for n in nodes if n['submit_partition']) - set(parts))
    if unlisted:
        warnings.append("'scontrol show partition' does not list %s: MaxCPUsPerNode and MaxTime "
                        "not checked there" % ', '.join(unlisted))
    unknown = sorted(exclude - set(records))
    if unknown:
        warnings.append('exclude list names nodes Slurm does not know: ' + compress_hostlist(unknown))
    if only is not None:
        unknown = sorted(only - set(records))
        if unknown:
            warnings.append('--only names nodes Slurm does not know: ' + compress_hostlist(unknown))
        excl_only = sorted(only & exclude)
        if excl_only:
            warnings.append('--only names excluded nodes, not tested: ' + compress_hostlist(excl_only))

    doc = {'schema': 1,
           'generated': datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
           'reservation': reservation,
           'only': args.only.strip() if only is not None else '',      # "" = a full run
           'settings': dict((k, v) for k, v in sorted(os.environ.items()) if k.startswith('SOL_')),
           'nodes': nodes}
    out = os.path.abspath(args.out)
    try:
        if not os.path.isdir(os.path.dirname(out)):
            os.makedirs(os.path.dirname(out))
        tmp = out + '.tmp'
        with open(tmp, 'w') as fp:
            json.dump(doc, fp, indent=1)
            fp.write('\n')
        os.replace(tmp, out)
    except (IOError, OSError) as e:
        sys.stderr.write('inventory.py: ERROR: cannot write %s: %s\n' % (out, e))
        return 2

    source = nodes_file + ' (captured, not live)' if nodes_file else 'scontrol'
    summary(nodes, ctx, args.out, source, res, warnings)
    if not nodes:
        sys.stderr.write('inventory.py: ERROR: no node lines in the scontrol output\n')
        return 1
    if not any(n['testable'] for n in nodes):
        sys.stderr.write('inventory.py: ERROR: no node is testable\n')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
