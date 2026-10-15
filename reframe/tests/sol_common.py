# sol_common.py - helpers shared by the Sol pre-release tests, inventory.py and report.py.
# Settings come from SOL_* environment variables (run_prerelease.sh exports site.conf as SOL_<KEY>).

import json
import os
import re
import shlex


def env(key, default=''):
    '''Value of SOL_<key>, or default when unset or empty.'''
    val = os.environ.get('SOL_' + key, '')
    return val if val.strip() else default


def env_int(key, default):
    try:
        return int(env(key, str(default)))
    except ValueError:
        return default


def env_float(key, default):
    try:
        return float(env(key, str(default)))
    except ValueError:
        return default


# --- hostlists -------------------------------------------------------------

def _split_top(expr):
    '''Split on commas that are not inside brackets.'''
    parts, depth, buf = [], 0, ''
    for ch in expr:
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(buf)
            buf = ''
        else:
            buf += ch
    parts.append(buf)
    return [p.strip() for p in parts if p.strip()]


def _expand_one(name):
    m = re.search(r'\[([^\]]*)\]', name)
    if not m:
        return [name]
    prefix, suffix = name[:m.start()], name[m.end():]
    out = []
    for rng in m.group(1).split(','):
        rng = rng.strip()
        if not rng:
            continue
        if '-' in rng:
            lo, hi = rng.split('-', 1)
            width = len(lo)
            for i in range(int(lo), int(hi) + 1):
                out.extend(_expand_one(prefix + str(i).zfill(width) + suffix))
        else:
            out.extend(_expand_one(prefix + rng + suffix))
    return out


def expand_hostlist(expr):
    '''"scg[007-008,013],sdg051" -> ["scg007", "scg008", "scg013", "sdg051"]'''
    if not expr:
        return []
    names = []
    for part in _split_top(expr):
        names.extend(_expand_one(part))
    return names


def compress_hostlist(names):
    '''["scg007", "scg008", "scg013", "sdg051"] -> "scg[007-008,013],sdg051"'''
    groups, plain = {}, []
    for n in sorted(set(names)):
        m = re.match(r'^(.*?)(\d+)$', n)
        if not m:
            plain.append(n)
            continue
        groups.setdefault((m.group(1), len(m.group(2))), []).append(int(m.group(2)))
    out = []
    for (prefix, width), nums in sorted(groups.items()):
        nums.sort()
        ranges, start, prev = [], nums[0], nums[0]
        for x in nums[1:] + [None]:
            if x is not None and x == prev + 1:
                prev = x
                continue
            if start == prev:
                ranges.append(str(start).zfill(width))
            else:
                ranges.append('%s-%s' % (str(start).zfill(width), str(prev).zfill(width)))
            if x is not None:
                start = prev = x
        if len(ranges) == 1 and '-' not in ranges[0]:
            out.append(prefix + ranges[0])
        else:
            out.append('%s[%s]' % (prefix, ','.join(ranges)))
    return ','.join(out + plain)


# --- inventory and settings ------------------------------------------------

def load_inventory(path=None):
    '''nodes.json as a dict; {"nodes": []} when the file is missing so --list still works.'''
    path = path or env('INVENTORY')
    if not path or not os.path.isfile(path):
        return {'nodes': [], 'settings': {}}
    with open(path) as fp:
        return json.load(fp)


def testable_nodes(accel=None, inventory=None):
    inv = inventory if inventory is not None else load_inventory()
    return [n for n in inv.get('nodes', [])
            if n.get('testable') and (accel is None or n.get('accel') == accel)]


def skip_nodes():
    '''Hostnames listed in SOL_SKIP_NODES_FILE (nodes that failed the node stage).'''
    path = env('SKIP_NODES_FILE')
    if not path or not os.path.isfile(path):
        return set()
    with open(path) as fp:
        return set(line.strip() for line in fp if line.strip() and not line.startswith('#'))


def node_modules(node):
    '''Module names for a node's architecture; "" means "not configured, skip that step".'''
    if node.get('arch') == 'aarch64':
        return {'gcc': env('MOD_GCC_AARCH64'),
                'cuda': env('MOD_CUDA_AARCH64'),
                'node_mpi': env('MOD_NODE_MPI_AARCH64')}
    return {'gcc': env('MOD_GCC', 'gcc/16.1.0'),
            'cuda': env('MOD_CUDA', 'cuda-13.2.1-gcc-13.2.0'),
            'node_mpi': env('MOD_NODE_MPI', 'openmpi/5.0.8-gcc-15.2.0')}


def mpi_candidates():
    '''[(key, module), ...] from MPI_CANDIDATES="openmpi=openmpi/5.0.8-gcc-15.2.0 mpich=..."'''
    spec = env('MPI_CANDIDATES', 'openmpi=openmpi/5.0.8-gcc-15.2.0 '
                                 'mpich=mpich/4.3.2-srun-gcc-15.2.0 hpcx=hpcx/2.25.1')
    out = []
    for item in spec.split():
        if '=' in item:
            key, mod = item.split('=', 1)
            out.append((key.strip(), mod.strip()))
    return out


DEFAULT_MPI_GROUPS = '''
cpu-zen3|public|sc[006-112]|64|no
cpu-zen4|general|scc[009-057]|32|no
gpu-a30|general|scg[007-008,013-014],sg[230-234]|9|yes
gpu-l40s|general|scg[015,017-019,022-023,025,027,029]|9|yes
'''


def mpi_groups():
    '''[{name, partition, hosts, coll_nodes, gpu}, ...] from MPI_GROUPS lines "name|partition|hostlist|n|yes/no".'''
    groups = []
    for line in env('MPI_GROUPS', DEFAULT_MPI_GROUPS).splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        f = [x.strip() for x in line.split('|')]
        if len(f) < 4:
            continue
        groups.append({'name': f[0], 'partition': f[1], 'hosts': expand_hostlist(f[2]),
                       'coll_nodes': int(f[3]),
                       'gpu': len(f) > 4 and f[4].lower() in ('yes', 'y', 'true', '1')})
    return groups


def usable_group_nodes(group, inventory=None, skip=None):
    '''Group hosts that are testable and not in the skip list, in hostlist order.'''
    inv = inventory if inventory is not None else load_inventory()
    skip = skip_nodes() if skip is None else skip
    by_name = dict((n['name'], n) for n in inv.get('nodes', []))
    return [h for h in group['hosts']
            if h in by_name and by_name[h].get('testable') and h not in skip]


# --- Slurm job options -----------------------------------------------------

def qos_for(partition):
    '''QOS from QOS_MAP="general=private public=public", falling back to QOS.'''
    for item in env('QOS_MAP').split():
        if '=' in item:
            part, qos = item.split('=', 1)
            if part == partition:
                return qos
    return env('QOS')


def common_job_options(partition):
    opts = ['--partition=%s' % partition]
    if env('RESERVATION'):
        opts.append('--reservation=%s' % env('RESERVATION'))
    if qos_for(partition):
        opts.append('--qos=%s' % qos_for(partition))
    if env('ACCOUNT'):
        opts.append('--account=%s' % env('ACCOUNT'))
    return opts


def node_job_options(node):
    '''sbatch options for a whole-node job (the node itself is set with job.pin_nodes).'''
    opts = common_job_options(node['submit_partition']) + ['--mem=0']
    if node.get('gres_request'):
        opts.append('--gres=%s' % node['gres_request'])
    return opts


# --- references and result lines ------------------------------------------

def load_references():
    path = env('REFERENCES')
    if not path or not os.path.isfile(path):
        return {}
    with open(path) as fp:
        return json.load(fp)


def parse_sol_line(line):
    '''"SOL_GPU_INFO uuid=GPU-1 name="NVIDIA L40S" mem_gib=44.4" -> ("SOL_GPU_INFO", {...})'''
    line = line.strip()
    if not line.startswith('SOL_'):
        return None, {}
    try:
        parts = shlex.split(line)
    except ValueError:
        parts = line.split()
    fields = {}
    for p in parts[1:]:
        if '=' in p:
            k, v = p.split('=', 1)
            fields[k] = v
    return parts[0], fields


def read_sol_lines(path):
    '''All SOL_* lines of a file as [(token, fields), ...]; [] if the file is missing.'''
    out = []
    if not path or not os.path.isfile(path):
        return out
    with open(path, errors='replace') as fp:
        for line in fp:
            tok, fields = parse_sol_line(line)
            if tok:
                out.append((tok, fields))
    return out
