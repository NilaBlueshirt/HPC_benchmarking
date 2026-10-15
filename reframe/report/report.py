#!/usr/bin/env python3
# report.py - turns one Sol pre-release run into a problem list (terminal summary + problems.csv),
# a long-term statistics record (metrics.csv, versions.csv) and a self-contained report.html.
#
# Usage:
#   report.py --run-dir results/<run> [--baseline results/<old-run>] [--parent results/<full-run>]
#             [--stage node|all] [--failed-nodes-out FILE] [--write-references FILE] [--quiet]
#
# Reads nodes.json, stage_node.json, stage_mpi.json and the job output of every test case in the run
# directory; missing pieces are reported, not fatal. --write-references only writes the medians file
# (it leaves the run's other outputs alone) and exits 1 when the run has no values for it.
# --parent names the full run an --only rerun is compared with where the rerun alone has too few nodes:
# OUTLIER for a group with fewer than 3 values uses the parent's median (its metrics.csv), and the NVIDIA
# driver of fewer than 3 nodes is compared with the parent's majority (its versions.csv). Kernel and slurmd
# need no parent: they are compared with every node's OS= and Version= that nodes.json recorded from scontrol.
# Exit code: 0 when no node has a problem of severity <= 5 (UNAVAILABLE .. TOO_SLOW) and no MPI-stage test
# failed; 1 otherwise (a failed MPI test or OSU build counts, an MPI group left out for too few nodes does
# not), or when there are no node-stage results at all (nothing was verified); 2 for usage errors. Nodes
# outside --only (selected=false in nodes.json) are only counted as "not selected". Standard library only.

import argparse
import csv
import datetime
import html
import json
import math
import os
import re
import statistics
import sys
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tests'))
import sol_common as sol  # noqa: E402

CATEGORIES = ['UNAVAILABLE', 'JOB_FAILED', 'WRONG_RESULT', 'MISMATCH', 'TOO_SLOW', 'OUTLIER',
              'SLOW_LINK', 'SLOWER_THAN_BASELINE', 'NOT_MEASURED']
SEVERITY = dict((c, i + 1) for i, c in enumerate(CATEGORIES))
EXIT_SEVERITY = 5
MPI_ORDER = ['openmpi', 'mpich', 'hpcx']
WIDTH = 120

# metric -> (check, unit, higher is better, label)
METRICS = {
    'copy': ('stream', 'MB/s', True, 'STREAM copy'),
    'scale': ('stream', 'MB/s', True, 'STREAM scale'),
    'add': ('stream', 'MB/s', True, 'STREAM add'),
    'triad': ('stream', 'MB/s', True, 'STREAM triad'),
    'mpi_init': ('mpi_hello', 's', False, 'MPI_Init time'),
    'barrier': ('mpi_hello', 'us', False, 'MPI_Barrier time'),
    'sgemm': ('gpu', 'TFLOP/s', True, 'SGEMM (FP32)'),
    'dgemm': ('gpu', 'TFLOP/s', True, 'DGEMM (FP64)'),
    'hbm': ('gpu', 'GB/s', True, 'GPU memory bandwidth'),
    'h2d': ('gpu', 'GB/s', True, 'PCIe host to device'),
    'd2h': ('gpu', 'GB/s', True, 'PCIe device to host'),
    'link_bw_median_node': ('pairwise', 'MB/s', True, 'Median link bandwidth per node'),
}
NODE_OUTLIER_METRICS = ['triad', 'mpi_init']
GPU_METRICS = ['sgemm', 'dgemm', 'hbm', 'h2d', 'd2h']
BASELINE_METRICS = NODE_OUTLIER_METRICS + ['barrier'] + GPU_METRICS + ['link_bw_median_node']
# MPI_Init and barrier times come from one cold launch per node (launch skew, PMIx, plugins loaded from the
# shared filesystem), so healthy nodes spread widely. Instead of OUTLIER_FRACTION and HISTORY_DROP they are
# flagged only above this factor of the group median (OUTLIER, mpi_init only) or of the baseline.
SLOW_FACTOR = {'mpi_init': 2.0, 'barrier': 2.0}
OSU_TESTS = {'SolOsuPt2Pt': 'pt2pt', 'SolOsuCollective': 'collective', 'SolOsuGpuDirect': 'gpudirect'}
TEST_ORDER = ['build', 'pairwise', 'pt2pt/bw', 'pt2pt/latency', 'collective/allreduce', 'collective/alltoall',
              'gpudirect/bw', 'gpudirect/latency']
UUID_RE = re.compile(r'\b(?:GPU|MIG)-[0-9a-fA-F-]{8,}\b')
METRIC_WORDS = {'link_bw_median_node': 'median link bandwidth'}
REFERENCE_SOURCE = ['']      # "run <id>" of references.json, when known
# OUTLIER needs this many values in a group; a smaller group of an --only rerun uses the parent run's median
MIN_GROUP = 3
# versions compared for MISMATCH: (SOL_IDENT field or "driver", label, the nodes.json key scontrol filled in)
VERSION_CHECKS = [('kernel', 'kernel', 'os_kernel'), ('slurmd', 'slurmd', 'slurmd_version'),
                  ('driver', 'NVIDIA driver', None)]
VERSIONS_HEADER = ['node', 'arch', 'kernel', 'slurmd', 'driver']


# --------------------------------------------------------------------------------------------
# small helpers

def num(s):
    '''float(s), or None for missing, unparsable, nan or inf values.'''
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def fmt(v, unit=''):
    '''Readable number: 561,234 / 85.1 / 2.51 / 0.403.'''
    if v is None:
        return 'n/a'
    a = abs(v)
    if a >= 1000:
        s = '{:,.0f}'.format(v)
    elif a >= 100:
        s = '%.0f' % v
    elif a >= 10:
        s = '%.1f' % v
    elif a >= 1:
        s = '%.2f' % v
    else:
        s = '%.3g' % v
    return s + (' ' + unit if unit else '')


def count(n):
    return '{:,}'.format(n)


def csv_num(v):
    if v is None:
        return ''
    if isinstance(v, (int, float)):
        return ('%.6g' % v) if abs(v) < 1e6 else ('%.1f' % v)
    return str(v)


def natkey(s):
    return [int(p) if p.isdigit() else p for p in re.split(r'(\d+)', s or '')]


def median(vals):
    return statistics.median(vals) if vals else None


def one_line(s):
    return ' '.join((s or '').split())


def hostlist(names):
    return sol.compress_hostlist(sorted(set(names), key=natkey))


def is_host(h):
    return bool(h) and h not in ('None', 'assigned', '(null)', 'None assigned')


def dev_label(uuid, pci=''):
    '''"GPU-6eed1390 (0000:65:00.0)": UUID prefix as nvidia-smi -L shows it, plus the PCI address.'''
    short = uuid[:12] if re.match(r'^(GPU|MIG)-', uuid or '') else (uuid or '?')
    return short + (' (%s)' % pci if pci else '')


# --------------------------------------------------------------------------------------------
# inputs

def load_json(path, warnings, what):
    if not os.path.isfile(path):
        warnings.append('%s not found in the run directory' % what)
        return None
    try:
        with open(path) as fp:
            return json.load(fp)
    except (OSError, ValueError) as e:
        warnings.append('%s unreadable: %s' % (what, e))
        return None


def load_stage(path, warnings, what):
    '''(session_info, [testcases]) keeping the highest run_index per test case name.'''
    rep = load_json(path, warnings, what)
    if not isinstance(rep, dict):
        return None, []
    best = {}
    for run in rep.get('runs') or []:
        for tc in run.get('testcases') or []:
            key = (tc.get('name') or tc.get('display_name') or tc.get('unique_name'), tc.get('system'),
                   tc.get('partition'), tc.get('environ'))
            ri = tc.get('run_index') if isinstance(tc.get('run_index'), int) else run.get('run_index', 0) or 0
            if key not in best or ri >= best[key][0]:
                best[key] = (ri, tc)
    return rep.get('session_info') or {}, [v[1] for v in best.values()]


def job_nodes(tc):
    nl = tc.get('job_nodelist')
    if isinstance(nl, str):
        nl = sol.expand_hostlist(nl) if nl.strip() else []
    return [h for h in (nl or []) if is_host(h)]


def stdout_path(tc, sinfo, run_dir):
    return job_file(tc, sinfo, run_dir, tc.get('job_stdout'))


def job_file(tc, sinfo, run_dir, fname):
    '''A file of the test case: outputdir for passed tests, stagedir otherwise (report-db.md section 5). The
    same path under <run-dir>/rfm is tried first, so a copied or moved run directory reads its own files; the
    absolute path recorded in the report is the fallback.'''
    if not fname:
        return None
    dirs = ['outputdir', 'stagedir'] if tc.get('result') == 'pass' else ['stagedir', 'outputdir']
    for key in dirs:
        d = tc.get(key)
        if not d or not isinstance(d, str):
            continue
        cands = []
        for pkey, sub in (('prefix_stage', 'stage'), ('prefix_output', 'output')):
            pre = (sinfo or {}).get(pkey)
            if pre and (d == pre or d.startswith(pre.rstrip('/') + '/')):
                cands.append(os.path.join(run_dir, 'rfm', sub, os.path.relpath(d, pre)))
        m = re.search(r'/rfm/(stage|output)/(.+)$', d) or re.search(r'.*/(stage|output)/(.+)$', d)
        if m:
            cands.append(os.path.join(run_dir, 'rfm', m.group(1), m.group(2)))
        cands.append(d)
        for c in cands:
            p = os.path.join(c, fname)
            if os.path.isfile(p):
                return p
    return None


def parse_node_output(path):
    res = {'found': bool(path), 'ident': None, 'driver': None, 'devices': None, 'srun': None, 'skips': [],
           'mpi': None, 'stream': None, 'stream_valid': True, 'gpus': {}, 'gpu_order': [], 'timeouts': [],
           'done': False, 'last': None}
    for tok, f in sol.read_sol_lines(path):
        if tok == 'SOL_IDENT':
            res['ident'] = f
        elif tok == 'SOL_DRIVER':
            res['driver'] = f.get('nvidia')
        elif tok == 'SOL_DEVICES':
            res['devices'] = f
        elif tok == 'SOL_SRUN':
            res['srun'] = f
        elif tok == 'SOL_SKIP':
            res['skips'].append((f.get('step') or '?', f.get('reason') or ''))
        elif tok == 'SOL_MPI_HELLO':
            res['mpi'] = {'mpi_init': num(f.get('init_s')), 'barrier': num(f.get('barrier_us'))}
        elif tok == 'SOL_STREAM':
            res['stream'] = dict((k, num(f.get(k))) for k in ('copy', 'scale', 'add', 'triad'))
            res['stream_valid'] = f.get('valid', '1') != '0'
        elif tok.startswith('SOL_GPU_') and f.get('uuid'):
            u = f['uuid']
            if u not in res['gpus']:
                res['gpus'][u] = {'uuid': u}
                res['gpu_order'].append(u)
            d = res['gpus'][u]
            if tok == 'SOL_GPU_INFO':
                d.update(name=f.get('name') or '', mem_gib=num(f.get('mem_gib')), pci=f.get('pci') or '')
            elif tok == 'SOL_GPU_PCIE':
                d.update(h2d=num(f.get('h2d_gbs')), d2h=num(f.get('d2h_gbs')))
            elif tok == 'SOL_GPU_HBM':
                d['hbm'] = num(f.get('hbm_gbs'))
            elif tok == 'SOL_GPU_GEMM':
                d.update(sgemm=num(f.get('sgemm_tflops')), dgemm=num(f.get('dgemm_tflops')),
                         errors=int(num(f.get('errors')) or 0))
        elif tok == 'SOL_TIMEOUT':
            res['timeouts'].append((f.get('step') or '?', f.get('seconds') or '?'))
        elif tok == 'SOL_DONE':
            res['done'] = True
        res['last'] = (tok, f)
    return res


def running_step(last):
    '''The node_check.sh step that was running after the last SOL line of a job that stopped early.'''
    tok, f = last or ('', {})
    order = ['ident', 'devices', 'srun', 'mpi_hello', 'stream', 'gpu']
    step = f.get('step')
    if tok == 'SOL_BUILD':
        return step if f.get('ok') == '1' or step not in order[:-1] else order[order.index(step) + 1]
    if tok in ('SOL_SKIP', 'SOL_TIMEOUT'):
        return step if step == 'gpu' or step not in order[:-1] else order[order.index(step) + 1]
    if tok.startswith('SOL_GPU_'):
        return 'gpu'
    return {'SOL_IDENT': 'devices', 'SOL_DRIVER': 'devices', 'SOL_DEVICES': 'srun', 'SOL_SRUN': 'mpi_hello',
            'SOL_MPI_HELLO': 'stream', 'SOL_STREAM': 'gpu'}.get(tok, '')


def parse_osu(path):
    '''OSU micro-benchmark output: ([(size, value), ...], unit). Takes the first table after "# Size".'''
    rows, unit, in_table = [], '', False
    if not path or not os.path.isfile(path):
        return rows, unit
    with open(path, errors='replace') as fp:
        for line in fp:
            s = line.strip()
            if s.startswith('# Size'):
                if rows:
                    break
                in_table = True
                unit = 'MB/s' if 'MB/s' in s else ('us' if '(us)' in s else '')
                continue
            if in_table:
                m = re.match(r'^(\d+)\s+([-+0-9.eE]+)', s)
                if m and num(m.group(2)) is not None:
                    rows.append((int(m.group(1)), num(m.group(2))))
    return rows, unit


def parse_pairs(path):
    pairs, summary = [], None
    for tok, f in sol.read_sol_lines(path):
        if tok == 'SOL_PAIR' and f.get('a') and f.get('b'):
            pairs.append((f['a'], f['b'], num(f.get('bibw_mbs')), num(f.get('lat_us'))))
        elif tok == 'SOL_PAIRBW':
            summary = f
    return pairs, summary


def apply_settings(inv):
    '''SOL_* values recorded in nodes.json fill in variables that are not set in this environment,
    so a report rerun by hand uses the run's own settings.'''
    for k, v in ((inv or {}).get('settings') or {}).items():
        if k.startswith('SOL_') and isinstance(v, str) and not os.environ.get(k, '').strip():
            os.environ[k] = v


def node_status(n):
    '''Where a nodes.json record is counted: not_selected (outside --only), excluded, unavailable, untestable
    (available but cannot run the whole-node test, untestable_reason) or testable. Records without the
    selected and untestable_reason keys count as selected, with no reason.'''
    if n.get('excluded'):
        return 'excluded'
    if not n.get('selected', True):
        return 'not_selected'
    if not n.get('available', True):
        return 'unavailable'
    if n.get('untestable_reason'):
        return 'untestable'
    if n.get('testable'):
        return 'testable'
    return 'not_selected'       # older nodes.json: available and not excluded, but outside --only


# --------------------------------------------------------------------------------------------
# problems

def problem(category, node, check, reason, group='', metric='', value=None, expected=None, **extra):
    p = {'severity': SEVERITY[category], 'category': category, 'node': node, 'group': group, 'check': check,
         'metric': metric, 'value': value, 'expected': expected, 'reason': one_line(reason)}
    p.update(extra)
    return p


def split_sanity(reason):
    '''"sanity error: [a] x; [b] y" -> [("a", "x"), ("b", "y")]; untagged text gets tag None.'''
    text = (reason or '').strip()
    text = re.sub(r'^sanity error:\s*', '', text)
    parts = re.split(r';\s*(?=\[[A-Za-z0-9_]+\])', text)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        m = re.match(r'^\[([A-Za-z0-9_]+)\]\s*(.*)$', p, re.S)
        out.append((m.group(1), one_line(m.group(2))) if m else (None, one_line(p)))
    return out


def clean_reason(reason):
    '''One readable line from a ReFrame fail_reason.'''
    r = (reason or '').strip()
    m = re.match(r'job blocked error: \[jobid=([^\]]*)\] job cancelled because it was blocked due to reason: '
                 r'(.*)$', r, re.S)
    if m:
        return 'blocked in the queue (%s); ReFrame cancelled job %s' % (one_line(m.group(2)), m.group(1))
    m = re.match(r"spawned process error: command '(.*?)' failed with exit code (-?\d+):(.*)$", r, re.S)
    if m:
        cmd, code, rest = m.group(1), m.group(2), m.group(3)
        msg = ''
        for block in ('stderr', 'stdout'):
            b = re.search(r'--- %s ---\n(.*?)\n?--- %s ---' % (block, block), rest, re.S)
            lines = [x.strip() for x in (b.group(1) if b else '').splitlines() if x.strip()]
            if lines:
                msg = lines[-1] if 'error' not in ' '.join(lines).lower() else \
                    next(x for x in lines if 'error' in x.lower())
                break
        return '%s failed (exit %s)%s' % (cmd, code, ': ' + msg if msg else '')
    m = re.match(r'job error: \[jobid=([^\]]*)\] (.*)$', r, re.S)
    if m:
        return 'job %s: %s' % (m.group(1), one_line(m.group(2)))
    return one_line(r) or 'no reason recorded'


def minutes(seconds):
    return '%d min' % round(seconds / 60.0) if seconds < 7200 else '%.1f h' % (seconds / 3600.0)


def duration(seconds):
    '''Seconds written as in site.conf: 3600 -> "60m", 14400 -> "4h", 90 -> "90s".'''
    s = int(round(seconds))
    if s >= 7200 and s % 3600 == 0:
        return '%dh' % (s // 3600)
    if s >= 60 and s % 60 == 0:
        return '%dm' % (s // 60)
    return '%ds' % s


def never_ran(tc, text, key, setting):
    '''Reason for a job that left no stdout: pending past its limit and cancelled, or ran without output.
    key names the limit (MAX_PENDING for the node stage, MPI_MAX_PENDING for the MPI stage); its value is the
    test case's own max_pending_time when the report has it, else setting.'''
    m = re.search(r'Slurm state ([A-Z_]+(?:[ ,][A-Z_]+)*)', text or '')
    state = ' (Slurm state %s)' % m.group(1) if m else ''
    fname = tc.get('job_stdout') or 'rfm_job.out'
    if job_nodes(tc):
        return 'job ran on %s but left no %s%s' % (hostlist(job_nodes(tc)), fname, state)
    sub, end = num(tc.get('job_submit_time')), num(tc.get('job_completion_time_unix'))
    cap = num(tc.get('max_pending_time'))
    limit = '%s=%s' % (key, duration(cap) if cap else setting)
    if sub is not None and end is not None and cap and end - sub >= 0.98 * cap:
        return ('never started: pending for %s, past %s, so ReFrame cancelled it%s'
                % (minutes(end - sub), limit, state))
    return 'never started: no %s%s; pending longer than %s, or cancelled' % (fname, state, limit)


def node_job_problems(tc, host, res, group, max_pending):
    '''JOB_FAILED / WRONG_RESULT rows for one SolNodeCheck test case (TOO_SLOW is handled separately).
    A [done] message is JOB_FAILED when the job never reached SOL_DONE, otherwise (a bad Slurm end state
    after a complete run) WRONG_RESULT like every other tagged message. max_pending is the MAX_PENDING
    setting, used when the test case does not record its max_pending_time.'''
    out = []
    result = tc.get('result')
    phase = tc.get('fail_phase')
    reason = tc.get('fail_reason') or ''
    if result == 'pass':
        return out
    if result in ('fail', 'abort', 'xfail', 'xpass') and phase == 'sanity':
        for tag, text in split_sanity(reason):
            low = text.lower()
            if (tag == 'done' and 'never ran' in low) or (tag is None and 'no such file' in low and not res['found']):
                out.append(problem('JOB_FAILED', host, tag or 'job', never_ran(tc, text, 'MAX_PENDING', max_pending),
                                   group))
            elif tag == 'done' and ('did not finish' in low or not res['done']):
                why = text
                step = running_step(res['last']) if res['found'] else ''
                if step:
                    tl = num(tc.get('time_limit'))
                    why += '; it stopped in the %s step%s' % (step, ' (time limit %s)' % minutes(tl)
                                                              if tl and 'TIMEOUT' in text else '')
                out.append(problem('JOB_FAILED', host, 'done', why, group))
            elif tag is None:
                out.append(problem('WRONG_RESULT', host, 'sanity', text, group))
            else:
                out.append(problem('WRONG_RESULT', host, tag, text, group))
        if not out:
            out.append(problem('WRONG_RESULT', host, 'sanity', clean_reason(reason), group))
    elif result in ('fail', 'abort') and phase == 'performance':
        return out
    elif result in ('fail', 'abort'):
        check = 'build' if phase in ('compile', 'compile_wait') else 'job'
        why = clean_reason(reason)
        if result == 'abort':
            why = 'aborted: ' + why
        out.append(problem('JOB_FAILED', host, check, '%s (ReFrame phase %s)' % (why, phase or '?'), group))
    elif result == 'skip':
        out.append(problem('JOB_FAILED', host, 'job', 'ReFrame skipped the test (the reason is only in the '
                                                       'ReFrame log)', group))
    elif result == 'fail_deps':
        out.append(problem('JOB_FAILED', host, 'job', 'not run: a dependency failed', group))
    else:
        out.append(problem('JOB_FAILED', host, 'job', 'unexpected ReFrame result %r' % result, group))
    return out


def outlier_limit(metric, med, of):
    '''Where OUTLIER starts for a group median: below OUTLIER_FRACTION x median (above median / OUTLIER_FRACTION
    when lower is better), or above SLOW_FACTOR x median for the start-up timings.'''
    if metric in SLOW_FACTOR:
        return SLOW_FACTOR[metric] * med
    return of * med if METRICS[metric][2] else med / of


def outlier_rows(groups, metric, kind, of, unit_label='nodes', untrusted=(), parent=None):
    '''groups: {group label: [(value, node, device, extra)]}; returns (rows, stats per group). Groups in
    `untrusted` (no real comparison group known) are shown but never checked. A group with fewer than MIN_GROUP
    values is compared with the parent run's median of the same group when the parent has at least MIN_GROUP
    values for it (parent: {'id': run id, 'medians': {(metric, group): (median, count)}}).'''
    check, unit, higher, _ = METRICS[metric]
    rows, stats = [], {}
    for g, pts in groups.items():
        vals = [p[0] for p in pts if p[0] is not None]
        med, n, src = median(vals), len(vals), ''
        ok = n >= MIN_GROUP and g not in untrusted
        if not ok and vals and g not in untrusted and parent:
            pmed, pn = parent['medians'].get((metric, g), (None, 0))
            if pmed is not None and pn >= MIN_GROUP:
                med, n, src, ok = pmed, pn, 'run %s' % parent['id'], True
        stats[g] = {'median': med, 'n': len(vals), 'checked': ok, 'src': src, 'src_n': n if src else 0}
        if med is None or not ok or med <= 0:
            continue
        limit = outlier_limit(metric, med, of)
        for v, node, dev, extra in pts:
            if v is None:
                continue
            bad = v < limit if higher else v > limit
            if not bad:
                continue
            pct = 100.0 * v / med
            what = '%s %s' % (METRIC_WORDS.get(metric, metric), fmt(v, unit))
            label = dev_label(dev, extra) if kind == 'gpu' else ''
            if label:
                what = '%s: %s' % (label, what)
            reason = '%s = %.0f%% of %s median %s%s (%s %s)' % (what, pct, g, fmt(med), ' from ' + src if src else '',
                                                                count(n), unit_label)
            rows.append(problem('OUTLIER', node, check, reason, g, metric, v, limit, device=dev, pct=pct,
                                label=label, src=src))
    return rows, stats


def real_version(v):
    '''node_check.sh prints NA when the tool failed, and nodes.json has "" when Slurm reported nothing.'''
    return bool(v) and v.strip() not in ('', 'NA')


def version_mismatches(run, A, parent=None):
    '''MISMATCH rows: each tested node's kernel, slurmd and NVIDIA driver against the majority of the nodes of
    the same architecture.
    - kernel, slurmd: every nodes.json record with a value (scontrol's OS= and Version=, the whole cluster); when
      nodes.json has no such values for the architecture, the nodes of this run, as before.
    - driver: this run's NVIDIA nodes when at least MIN_GROUP of them reported one; otherwise the majority in the
      parent run's versions.csv; otherwise no check.'''
    nodes = run.nodes
    out = []
    for attr, label, inv_key in VERSION_CHECKS:
        measured = {}
        for h, res in A['node_res'].items():
            if attr == 'driver':
                if (nodes.get(h) or {}).get('accel', 'nvidia') != 'nvidia':
                    continue
                val = res['driver']
            else:
                val = (res['ident'] or {}).get(attr)
            if not real_version(val):
                continue
            arch = (res['ident'] or {}).get('arch') or (nodes.get(h) or {}).get('arch') or '?'
            measured.setdefault(arch, {})[h] = val
        for arch, vals in sorted(measured.items()):
            ref, where = list(vals.values()), 'this run'
            if inv_key:
                known = [n[inv_key] for n in nodes.values() if n.get('arch') == arch and real_version(n.get(inv_key))]
                if known:
                    ref, where = known, 'from scontrol'
            elif len(vals) < MIN_GROUP:
                known = [r[4] for r in (parent or {}).get('versions', []) if r[1] == arch and real_version(r[4])]
                if not known:
                    continue            # too few NVIDIA nodes here and no parent run to compare with
                ref, where = known, 'run %s' % parent['id']
            counts = {}
            for v in ref:
                counts[v] = counts.get(v, 0) + 1
            ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            top, top_n = ranked[0]
            tie = len(ranked) > 1 and ranked[1][1] == top_n
            for h, v in sorted(vals.items(), key=lambda kv: natkey(kv[0])):
                if not tie and v == top:
                    continue
                if tie:
                    reason = '%s %s; no majority among %d %s nodes (%s: %s)' % (
                        label, v, len(ref), arch, where, ', '.join('%s on %d' % kv for kv in ranked))
                else:
                    reason = '%s %s; majority of %s nodes run %s (%s, %d of %d %snodes)' % (
                        label, v, arch, top, where, top_n, len(ref), arch + ' ' if where == 'from scontrol' else '')
                out.append(problem('MISMATCH', h, attr, reason, run.cpu_group(h), attr, v, top))
    return out


def version_rows(run, A):
    '''versions.csv: [node, arch, kernel, slurmd, driver] as each node's SOL_IDENT and SOL_DRIVER lines reported
    them (NA = the tool failed on the node; an empty driver = no SOL_DRIVER line, not an NVIDIA node).'''
    rows = []
    for h in sorted(A['node_res'], key=natkey):
        res = A['node_res'][h]
        ident = res['ident'] or {}
        if not ident and not res['driver']:
            continue
        rows.append([h, ident.get('arch') or (run.nodes.get(h) or {}).get('arch') or '', ident.get('kernel') or '',
                     ident.get('slurmd') or '', res['driver'] or ''])
    return rows


# --------------------------------------------------------------------------------------------
# the run

class Run:
    def __init__(self, run_dir, stage='all'):
        self.dir = run_dir
        self.stage = stage
        self.warnings = []
        inv = load_json(os.path.join(run_dir, 'nodes.json'), self.warnings, 'inventory nodes.json')
        self.inv = inv if isinstance(inv, dict) else {'nodes': [], 'settings': {}}
        self.nodes = dict((n.get('name'), n) for n in self.inv.get('nodes') or [] if n.get('name'))
        self.node_info, self.node_tcs = load_stage(os.path.join(run_dir, 'stage_node.json'), self.warnings,
                                                   'node stage report stage_node.json')
        self.have_node_stage = self.node_info is not None
        self.mpi_info, self.mpi_tcs = (None, [])
        self.have_mpi_stage = False
        if stage == 'all':
            p = os.path.join(run_dir, 'stage_mpi.json')
            if os.path.isfile(p):
                self.mpi_info, self.mpi_tcs = load_stage(p, self.warnings, 'MPI stage report stage_mpi.json')
                self.have_mpi_stage = self.mpi_info is not None
        info = self.node_info or self.mpi_info or {}
        self.run_id = info.get('run') or os.path.basename(os.path.realpath(run_dir.rstrip('/')))
        self.idents = {}
        self.tag_groups = {}

    def cpu_group(self, host, ident=None):
        '''The node's comparison group. Without an inventory entry: the cpu_group tag of its test case, else a
        hardware signature from SOL_IDENT, so nodes of different hardware are never compared.'''
        g = (self.nodes.get(host) or {}).get('cpu_group') or self.tag_groups.get(host)
        if g:
            return g
        ident = ident if ident is not None else self.idents.get(host)
        if ident:
            mem = num(ident.get('mem_kb'))
            return 'unknown-%s-%scpu-%sG' % (ident.get('arch', '?'), ident.get('nproc', '?'),
                                             -(-int(mem) // (1024 * 1024 * 64)) * 64 if mem else '?')
        return 'unknown'


def analyze(run, of, parent=None):
    '''Everything derived from one run: problems, metrics and chart data. parent (see load_parent) supplies the
    medians and driver versions for groups too small in this run.'''
    A = {'problems': [], 'metrics': [], 'node_res': {}, 'node_tc': {}, 'gpu_pool': {}, 'gpu_pci': {}, 'pairs': {},
         'osu': {}, 'mpi_rows': [], 'mpi_groups': [], 'strips': {}, 'of': of, 'flag_rows': [], 'parent': parent}
    max_pending = sol.env('MAX_PENDING', '30m')
    nodes = run.nodes
    P = A['problems']

    # from the inventory: UNAVAILABLE, and NOT_MEASURED for nodes that cannot run the whole-node test.
    # Nodes outside --only get no rows.
    for h, n in sorted(nodes.items(), key=lambda kv: natkey(kv[0])):
        status = node_status(n)
        if status == 'unavailable':
            P.append(problem('UNAVAILABLE', h, 'slurm', n.get('unavailable_reason') or 'not available',
                             n.get('cpu_group', '')))
        elif status == 'untestable':
            P.append(problem('NOT_MEASURED', h, 'slurm', n['untestable_reason'], n.get('cpu_group', '')))

    # node stage
    for tc in run.node_tcs:
        host = tc.get('node')
        if not isinstance(host, str) or tc.get('fixture'):
            continue
        path = stdout_path(tc, run.node_info, run.dir)
        res = parse_node_output(path)
        A['node_res'][host] = res
        A['node_tc'][host] = tc
        if res['ident']:
            run.idents[host] = res['ident']
        for t in tc.get('tags') or []:
            if isinstance(t, str) and re.match(r'^[a-z0-9_]+-\d+c-\d+g$', t):
                run.tag_groups[host] = t[:-1] + 'G'       # the tag is the lower-cased cpu_group
        grp = run.cpu_group(host)
        P.extend(node_job_problems(tc, host, res, grp, max_pending))
        for key, e in (tc.get('perfvalues') or {}).items():
            if not isinstance(e, list) or len(e) < 6 or e[5] != 'fail':
                continue
            var = key.split(':')[-1]
            val, ref, low, high, unit = num(e[0]), num(e[1]), num(e[2]), num(e[3]), e[4] or ''
            lower = ref * (1 + low) if ref is not None and low is not None else None
            upper = ref * (1 + high) if ref is not None and high is not None else None
            base = var[:-4] if var.endswith('_min') else var
            check = METRICS.get(base, ('perf',))[0]
            src = ' (%s)' % REFERENCE_SOURCE[0] if REFERENCE_SOURCE[0] else ''
            if val is not None and lower is not None and val < lower:
                reason = '%s %s is below %s = %.0f%% of the reference %s%s' % (
                    var, fmt(val, unit), fmt(lower), 100 * (1 + low), fmt(ref), src)
                bound = lower
            elif val is not None and upper is not None and val > upper:
                reason = '%s %s is above %s = %.0f%% of the reference %s%s' % (
                    var, fmt(val, unit), fmt(upper), 100 * (1 + high), fmt(ref), src)
                bound = upper
            else:
                reason = '%s %s failed its reference %s%s' % (var, fmt(val, unit), fmt(ref), src)
                bound = lower if lower is not None else upper
            P.append(problem('TOO_SLOW', host, check, reason, grp, var, val, bound, base_metric=base))
        for step, why in res['skips']:
            P.append(problem('NOT_MEASURED', host, step, why or 'step skipped (no reason given)', grp))
        if not res['found'] and (tc.get('result') == 'pass' or tc.get('fail_phase') == 'performance'):
            P.append(problem('NOT_MEASURED', host, 'output', 'no output (%s not found; job never ran or files were '
                                                             'cleaned): no metrics for this node'
                             % (tc.get('job_stdout') or 'stdout'), grp))
    if run.have_node_stage:
        seen = set(A['node_tc'])
        for h, n in sorted(nodes.items(), key=lambda kv: natkey(kv[0])):
            if n.get('testable') and h not in seen:
                P.append(problem('JOB_FAILED', h, 'job', 'no test case for this node in stage_node.json '
                                                         '(not run or filtered out)', n.get('cpu_group', '')))

    # MISMATCH: kernel, slurmd, NVIDIA driver against the majority of the same arch
    P.extend(version_mismatches(run, A, parent))

    # node metrics, CPU outliers
    cpu_groups = {}
    for h, res in A['node_res'].items():
        grp = run.cpu_group(h)
        vals = {}
        if res['stream'] and res['stream_valid']:
            vals.update(res['stream'])
        if res['mpi']:
            vals.update(res['mpi'])
        for m, v in vals.items():
            if v is None:
                continue
            A['metrics'].append((h, grp, '', '', METRICS[m][0], m, v, METRICS[m][1]))
            if m in NODE_OUTLIER_METRICS:
                cpu_groups.setdefault(m, {}).setdefault(grp, []).append((v, h, '', ''))
    guessed = set(g for d in cpu_groups.values() for g in d if g.startswith('unknown'))
    if guessed:
        run.warnings.append('%d nodes have no CPU group (no nodes.json entry and no cpu_group tag): not checked '
                            'for CPU outliers' % len(set(h for d in cpu_groups.values() for g in guessed
                                                         for _, h, _, _ in d.get(g, []))))
    for m in NODE_OUTLIER_METRICS:
        rows, stats = outlier_rows(cpu_groups.get(m, {}), m, 'node', of, untrusted=guessed, parent=parent)
        P.extend(rows)
        A['strips'][m] = (cpu_groups.get(m, {}), stats)

    # GPU devices, pooled by name + memory
    pools = {}
    for h, res in A['node_res'].items():
        gm = (nodes.get(h) or {}).get('gpu_model') or 'unknown'
        for u in res['gpu_order']:
            d = res['gpus'][u]
            if not d.get('name'):
                continue
            mem = d.get('mem_gib')
            pool = '%s (%s GiB)' % (d['name'], int(round(mem)) if mem is not None else '?')
            A['gpu_pool'][u] = pool
            A['gpu_pci'][u] = d.get('pci', '')
            for m in GPU_METRICS:
                v = d.get(m)
                if v is None:
                    continue
                A['metrics'].append((h, 'gpu:' + gm, u, pool, 'gpu', m, v, METRICS[m][1]))
                pools.setdefault(m, {}).setdefault(pool, []).append((v, h, u, d.get('pci', '')))
    for m in GPU_METRICS:
        rows, stats = outlier_rows(pools.get(m, {}), m, 'gpu', of, 'devices', parent=parent)
        P.extend(rows)
        A['strips'][m] = (pools.get(m, {}), stats)

    # MPI stage
    if run.stage == 'all':
        analyze_mpi(run, A, of)
    A['flag_rows'] = [dict(p) for p in P if p['category'] == 'OUTLIER']
    return A


def build_reason(tc, run):
    '''A failed build fixture: ReFrame's reason plus the first error line of rfm_build.err.'''
    why = clean_reason(tc.get('fail_reason')) if tc.get('result') == 'fail' else str(tc.get('result'))
    path = job_file(tc, run.mpi_info, run.dir, tc.get('build_stderr') or 'rfm_build.err')
    lines = []
    if path:
        try:
            with open(path, errors='replace') as fp:
                lines = [x.strip() for x in fp if x.strip()]
        except OSError:
            pass
    hit = next((x for x in lines if re.search(r'\berror\b', x, re.I)), lines[-1] if lines else '')
    return why + (' (%s: %s)' % (os.path.basename(path), hit[:200]) if hit else '')


def mpi_key(tc):
    cls = (tc.get('name') or '').split(' ')[0]
    bench = tc.get('bench') if isinstance(tc.get('bench'), str) else ''
    if cls == 'SolPairwiseBW':
        test = 'pairwise'
    elif cls in OSU_TESTS:
        test = '%s/%s' % (OSU_TESTS[cls], bench or '?')
    elif tc.get('fixture'):
        test = 'build'
    else:
        test = cls
    mpi = tc.get('mpi') if isinstance(tc.get('mpi'), str) else (tc.get('environ') or '-')
    group = tc.get('group') if isinstance(tc.get('group'), str) else ''
    return cls, group, mpi, test


def analyze_mpi(run, A, of):
    P = A['problems']
    groups = sol.mpi_groups()
    skip_file = sol.env('SKIP_NODES_FILE') or os.path.join(run.dir, 'skip_mpi.txt')
    skip = set()
    if os.path.isfile(skip_file):
        with open(skip_file) as fp:
            skip = set(x.strip() for x in fp if x.strip() and not x.startswith('#'))
    build_fail = {}
    for tc in run.mpi_tcs:
        if tc.get('fixture') and tc.get('result') != 'pass':
            build_fail[tc.get('environ')] = build_reason(tc, run)
    for tc in run.mpi_tcs:
        cls, group, mpi, test = mpi_key(tc)
        result = tc.get('result')
        path = stdout_path(tc, run.mpi_info, run.dir)
        perf = []
        for key, e in (tc.get('perfvalues') or {}).items():
            if isinstance(e, list) and len(e) >= 5 and num(e[0]) is not None:
                perf.append((key.split(':')[-1], num(e[0]), e[4] or ''))
        nodes_used = job_nodes(tc)
        row = {'group': group or '-', 'mpi': mpi, 'test': test, 'result': result, 'perf': perf,
               'nodes': nodes_used, 'num_tasks': tc.get('num_tasks'), 'tpn': tc.get('num_tasks_per_node'),
               'reason': ''}
        if tc.get('fixture'):
            if result == 'pass':
                continue
            row['group'] = '-'
            row['reason'] = build_fail.get(tc.get('environ')) or str(result)
            P.append(problem('JOB_FAILED', '', '%s:build' % tc.get('environ'), 'OSU build for %s failed: %s'
                             % (tc.get('environ'), row['reason']), ''))
            A['mpi_rows'].append(row)
            continue
        if result == 'pass' and not path and cls in ('SolPairwiseBW',) + tuple(OSU_TESTS):
            row['nodata'] = True
        if result != 'pass':
            phase = tc.get('fail_phase')
            if result == 'fail_deps':
                row['reason'] = 'not run: ' + ('OSU build for %s failed' % tc.get('environ')
                                               if tc.get('environ') in build_fail else 'a dependency failed')
                cat = 'JOB_FAILED'
            elif result == 'skip':
                row['reason'] = 'skipped by ReFrame (reason only in the ReFrame log)'
                cat = 'JOB_FAILED'
            elif phase == 'sanity' and re.search(r'no such file', tc.get('fail_reason') or '', re.I) and not path:
                row['reason'] = never_ran(tc, tc.get('fail_reason'), 'MPI_MAX_PENDING',
                                          sol.env('MPI_MAX_PENDING', '4h'))
                cat = 'JOB_FAILED'
            elif phase == 'sanity':
                row['reason'] = 'wrong result: ' + '; '.join(t if not g else '[%s] %s' % (g, t)
                                                             for g, t in split_sanity(tc.get('fail_reason')))
                cat = 'WRONG_RESULT'
            elif phase == 'performance':
                row['reason'] = clean_reason(tc.get('fail_reason'))
                cat = 'TOO_SLOW'
            else:
                row['reason'] = '%s (ReFrame phase %s)' % (clean_reason(tc.get('fail_reason')), phase or '?')
                cat = 'JOB_FAILED'
            P.append(problem(cat, '', '%s:%s' % (mpi, test), row['reason'], group))
        A['mpi_rows'].append(row)
        # data
        if cls == 'SolPairwiseBW' and group:
            pairs, summary = parse_pairs(path)
            if pairs:
                A['pairs'][group] = {'pairs': pairs, 'summary': summary}
        elif cls in OSU_TESTS and group:
            rows, unit = parse_osu(path)
            if rows:
                bench = test.split('/')[-1]
                check = 'osu_%s' % bench + ('_cuda' if cls == 'SolOsuGpuDirect' else '')
                A['osu'].setdefault(group, {})[(test, mpi)] = {'rows': rows, 'unit': unit, 'ok': result == 'pass'}
                for s, v in rows:
                    A['metrics'].append(('', group, mpi, '', check, str(s), v, unit))

    # pairwise: per-node median link bandwidth, node outliers and slow links
    pair_node_groups = {}
    for group, d in A['pairs'].items():
        per = {}
        for a, b, bw, lat in d['pairs']:
            if bw is None:
                continue
            per.setdefault(a, []).append(bw)
            per.setdefault(b, []).append(bw)
        node_med = dict((h, median(v)) for h, v in per.items())
        d['node_median'] = node_med
        d['median'] = median([bw for _, _, bw, _ in d['pairs'] if bw is not None])
        pair_node_groups[group] = [(v, h, '', '') for h, v in sorted(node_med.items(), key=lambda kv: natkey(kv[0]))]
        for h, v in node_med.items():
            A['metrics'].append((h, group, '', '', 'pairwise', 'link_bw_median_node', v, 'MB/s'))
    rows, stats = outlier_rows(pair_node_groups, 'link_bw_median_node', 'node', of, parent=A.get('parent'))
    P.extend(rows)
    A['strips']['link_bw_median_node'] = (pair_node_groups, stats)
    slow_nodes = set((r['group'], r['node']) for r in rows)
    for group, d in A['pairs'].items():
        med = d['median']
        if med is None or len(d['node_median']) < 3:
            continue
        for a, b, bw, lat in d['pairs']:
            if bw is None or bw >= of * med or (group, a) in slow_nodes or (group, b) in slow_nodes:
                continue
            reason = ('link %s <-> %s %s = %.0f%% of %s pair median %s (%s pairs); the pair was measured alone (no '
                      'contention) and both nodes are normal on their other links: suspect cable or switch port'
                      % (a, b, fmt(bw, 'MB/s'), 100 * bw / med, group, fmt(med), count(len(d['pairs']))))
            P.append(problem('SLOW_LINK', '%s+%s' % (a, b), 'pairwise', reason, group, 'bibw', bw, of * med,
                             hosts=[a, b], lat=lat))
    # groups left out of the MPI stage
    if run.have_mpi_stage:
        tested = set(r['group'] for r in A['mpi_rows'])
        for g in groups:
            usable = sol.usable_group_nodes(g, inventory=run.inv, skip=skip)
            info = {'name': g['name'], 'hosts': g['hosts'], 'usable': usable, 'coll_nodes': g['coll_nodes']}
            A['mpi_groups'].append(info)
            if len(usable) >= 2 or g['name'] in tested:
                continue
            not_selected = [h for h in g['hosts'] if h in run.nodes and node_status(run.nodes[h]) == 'not_selected']
            if not_selected and len(not_selected) == len(g['hosts']):
                continue            # an --only run that selected none of the group's nodes
            not_testable = [h for h in g['hosts'] if not (run.nodes.get(h) or {}).get('testable')
                            and h not in not_selected]
            failed = [h for h in g['hosts'] if h in skip]
            why = '%d usable node%s of %d (needs 2)' % (len(usable), '' if len(usable) == 1 else 's', len(g['hosts']))
            if usable:
                why += ': %s' % hostlist(usable)
            if failed:
                why += '; failed the node stage: %s' % hostlist(failed)
            if not_testable:
                why += '; not testable: %s' % hostlist(not_testable)
            if not_selected:
                why += '; not selected (--only): %s' % hostlist(not_selected)
            P.append(problem('NOT_MEASURED', '', 'mpi', 'MPI group left out: ' + why, g['name']))
            A['mpi_rows'].append({'group': g['name'], 'mpi': '-', 'test': 'all', 'result': 'not run', 'perf': [],
                                  'nodes': [], 'reason': why, 'num_tasks': None, 'tpn': None})


# --------------------------------------------------------------------------------------------
# baseline

def baseline_metrics(base_dir, of):
    '''{(kind, key, metric): value} plus the baseline run id. Prefers the baseline's metrics.csv
    (the long-term record); otherwise analyses the baseline run directory.'''
    out = {}
    path = os.path.join(base_dir, 'metrics.csv')
    rid = None
    rows = []
    if os.path.isfile(path):
        with open(path, newline='') as fp:
            for r in csv.DictReader(fp):
                rows.append((r.get('node', ''), r.get('group', ''), r.get('device', ''), r.get('device_name', ''),
                             r.get('check', ''), r.get('metric', ''), num(r.get('value')), r.get('unit', '')))
                rid = rid or r.get('run_id')
    else:
        brun = Run(base_dir, 'all')
        rid = brun.run_id
        rows = analyze(brun, of)['metrics']
    for node, group, dev, dname, check, metric, v, unit in rows:
        if v is None or metric not in METRICS:
            continue
        if check == 'gpu' and dev:
            out[('gpu', dev, metric)] = v
        elif check == 'pairwise' and node:
            out[('pair', group + '/' + node, metric)] = v
        elif node and not dev:
            out[('node', node, metric)] = v
    return out, rid or os.path.basename(os.path.realpath(base_dir.rstrip('/')))


def read_csv(path):
    with open(path, newline='') as fp:
        return list(csv.DictReader(fp))


def load_parent(pdir, of):
    '''The full run an --only rerun is compared with: {'id', 'medians': {(metric, group): (median, count)},
    'versions': [(node, arch, kernel, slurmd, driver)]}, from its metrics.csv and versions.csv. When either file
    is missing (its report never ran, or ran before versions.csv existed), the values are analysed from the
    run directory itself.'''
    mpath, vpath = os.path.join(pdir, 'metrics.csv'), os.path.join(pdir, 'versions.csv')
    rid, metrics, versions = None, None, None
    if os.path.isfile(mpath):
        metrics = []
        for r in read_csv(mpath):
            metrics.append((r.get('node', ''), r.get('group', ''), r.get('device', ''), r.get('device_name', ''),
                            r.get('check', ''), r.get('metric', ''), num(r.get('value')), r.get('unit', '')))
            rid = rid or r.get('run_id')
    if os.path.isfile(vpath):
        versions = [tuple(r.get(k) or '' for k in VERSIONS_HEADER) for r in read_csv(vpath)]
    if metrics is None or versions is None:
        prun = Run(pdir, 'all')
        PA = analyze(prun, of)
        rid = rid or prun.run_id
        metrics = PA['metrics'] if metrics is None else metrics
        versions = [tuple(r) for r in version_rows(prun, PA)] if versions is None else versions
    vals = {}
    for node, group, dev, dname, check, metric, v, unit in metrics:
        if v is None:
            continue
        if metric in NODE_OUTLIER_METRICS and node and not dev:
            key = (metric, group)                       # CPU group
        elif metric in GPU_METRICS and check == 'gpu' and dev:
            key = (metric, dname)                       # GPU pool: device name and memory
        elif metric == 'link_bw_median_node' and node:
            key = (metric, group)                       # MPI group
        else:
            continue
        vals.setdefault(key, []).append(v)
    return {'id': rid or os.path.basename(os.path.realpath(pdir.rstrip('/'))),
            'medians': dict((k, (median(v), len(v))) for k, v in vals.items()), 'versions': versions}


def baseline_drop(metric, hd):
    '''Tolerated slow-down against the baseline on the change scale (negative = slower; 2x the time = -50%):
    HISTORY_DROP, or up to SLOW_FACTOR times the baseline for the start-up timings.'''
    return 1 - 1 / SLOW_FACTOR[metric] if metric in SLOW_FACTOR else hd


def baseline_compare(A, base, base_id, hd):
    '''SLOWER_THAN_BASELINE rows plus chart data {metric: [(pct change, node, device, now, before, group)]}.'''
    rows, chart = [], {}
    keyed = []
    pci = {}
    for res in A['node_res'].values():
        for u, d in res['gpus'].items():
            pci[u] = d.get('pci', '')
    for node, group, dev, dname, check, metric, v, unit in A['metrics']:
        if metric not in BASELINE_METRICS:
            continue
        if check == 'gpu':
            key, label_group = ('gpu', dev, metric), dname or group
        elif check == 'pairwise':
            key, label_group = ('pair', group + '/' + node, metric), group
        else:
            key, label_group = ('node', node, metric), group
        keyed.append((key, node, dev, metric, v, label_group))
    for key, node, dev, metric, v, label_group in keyed:
        b = base.get(key)
        if b is None or b <= 0 or v is None or v <= 0:
            continue
        higher = METRICS[metric][2]
        change = (v / b - 1) if higher else (b / v - 1)       # negative = slower
        chart.setdefault(metric, []).append((100 * change, node, dev, v, b, label_group))
        drop = baseline_drop(metric, hd)
        if change < -drop:
            unit = METRICS[metric][1]
            label = dev_label(dev, pci.get(dev, '')) if dev else ''
            what = (label + ': ' if label else '') + '%s %s' % (METRIC_WORDS.get(metric, metric), fmt(v, unit))
            if higher:
                reason = '%s is %.0f%% below baseline %s (run %s)' % (what, -100 * change, fmt(b), base_id)
                expected = b * (1 - drop)
            else:
                reason = '%s is %sx the baseline %s (run %s)' % (what, fmt(v / b), fmt(b), base_id)
                expected = b / (1 - drop)
            rows.append(problem('SLOWER_THAN_BASELINE', node, METRICS[metric][0], reason, label_group, metric, v,
                                expected, device=dev, change=100 * change, label=label))
    return rows, chart


# --------------------------------------------------------------------------------------------
# merging, systemic issues, ordering

def merge_perf_rows(P):
    '''One row per slow (node, metric): TOO_SLOW absorbs the OUTLIER and SLOWER_THAN_BASELINE rows of the same
    node and metric; OUTLIER absorbs SLOWER_THAN_BASELINE of the same node, device and metric. The absorbed
    facts are appended to the reason.'''
    too_slow, outliers = {}, {}
    for p in P:
        if p['category'] == 'TOO_SLOW' and p['node']:
            too_slow.setdefault((p['node'], p.get('base_metric') or p['metric']), p)
        elif p['category'] == 'OUTLIER' and p['node'] and '+' not in p['node']:
            outliers[(p['node'], p.get('device', ''), p['metric'])] = p
    keep = []
    for p in P:
        if p['category'] in ('OUTLIER', 'SLOWER_THAN_BASELINE') and p['node'] and '+' not in p['node']:
            t = too_slow.get((p['node'], p['metric']))
            if t is not None:
                t.setdefault('_notes', []).append((p.get('label', ''), short_note(p)))
                continue
            if p['category'] == 'SLOWER_THAN_BASELINE':
                o = outliers.get((p['node'], p.get('device', ''), p['metric']))
                if o is not None:
                    o.setdefault('_notes', []).append(('', short_note(p)))
                    continue
        keep.append(p)
    for p in keep:
        notes = p.pop('_notes', None)
        if not notes:
            continue
        by_label = {}
        for label, note in notes:
            by_label.setdefault(label, []).append(note)
        parts = [(label + ': ' if label else 'also ') + ', '.join(ns) for label, ns in by_label.items()]
        p['reason'] += '; ' + '; '.join(parts)
    return keep


def short_note(p):
    if p['category'] == 'OUTLIER':
        return '%.0f%% of %s median%s' % (p.get('pct', 0), p['group'], ' from ' + p['src'] if p.get('src') else '')
    if p['metric'] in METRICS and not METRICS[p['metric']][2]:     # a time: change = baseline / now - 1
        return '%sx the baseline' % fmt(1 / (1 + p.get('change', 0) / 100.0))
    return '%+.0f%% vs baseline' % p.get('change', 0)


def normalize(reason, names):
    s = UUID_RE.sub('<uuid>', reason)
    s = re.sub(r'\b[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.\d\b', '<pci>', s)
    s = re.sub(r'\b[A-Za-z][A-Za-z_-]*\d+[a-z]?\b', lambda m: '<host>' if m.group(0) in names else m.group(0), s)
    s = re.sub(r'\d+(?:[.,]\d+)*', '#', s)
    return s


def find_systemic(P, run):
    '''Same (category, check, normalized reason) on >= 5 nodes, or on >= 50% (and >= 2 nodes) of a group
    with >= 2 nodes. Groups are CPU groups, GPU models and MPI groups.'''
    names = set(run.nodes) | set(p['node'] for p in P if p['node'])
    for p in P:
        for h in p.get('hosts') or []:
            names.add(h)
    members = {}
    for h, n in run.nodes.items():
        if n.get('excluded'):
            continue
        members.setdefault('cpu:' + (n.get('cpu_group') or '?'), set()).add(h)
        if n.get('gpu_model'):
            members.setdefault('gpu:' + n['gpu_model'], set()).add(h)
    for g in sol.mpi_groups():
        members['mpi:' + g['name']] = set(h for h in g['hosts'] if h in run.nodes and not run.nodes[h].get('excluded'))
    buckets = {}
    for i, p in enumerate(P):
        if not p['node']:
            continue
        key = (p['category'], p['check'], normalize(p['reason'], names))
        buckets.setdefault(key, []).append(i)
    issues = []
    for key, idx in buckets.items():
        rows = [P[i] for i in idx]
        if key[0] == 'SLOW_LINK':
            hosts = sorted(set(h for r in rows for h in r.get('hosts') or []), key=natkey)
            count = len(rows)
            systemic = count >= 5
        else:
            hosts = sorted(set(r['node'] for r in rows), key=natkey)
            count = len(hosts)
            systemic = count >= 5
            if not systemic and count >= 2:
                hs = set(hosts)
                for g, mem in members.items():
                    if len(mem) >= 2 and len(hs & mem) * 2 >= len(mem):
                        systemic = True
                        break
        if not systemic:
            continue
        reasons = set(r['reason'] for r in rows)
        rep = rows[0]['reason'] if len(reasons) == 1 else 'e.g. %s: %s' % (rows[0]['node'], rows[0]['reason'])
        issues.append({'category': key[0], 'check': key[1], 'severity': SEVERITY[key[0]], 'hosts': hosts,
                       'count': count, 'reason': rep, 'items': rows})
        for i in idx:
            P[i]['systemic'] = True
    issues.sort(key=lambda s: (s['severity'], -s['count'], s['check']))
    return issues


def order_rows(P):
    '''Node rows worst node first (by the node's worst severity), then each node's rows by severity.'''
    worst = {}
    for p in P:
        worst[p['node']] = min(worst.get(p['node'], 99), p['severity'])
    return sorted(P, key=lambda p: (0 if p['node'] else 1, worst[p['node']], natkey(p['node']), p['severity'],
                                    p['check'], p.get('device', '') or '', p['metric']))


# --------------------------------------------------------------------------------------------
# outputs

def write_csv(path, header, rows):
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)


def write_references(path, run, A):
    '''Medians per CPU group (triad, copy) and per GPU model (per-node minimum over devices), from the nodes
    whose check passed sanity (passed, or failed only on performance).'''
    cpu, gpu = {}, {}
    for h, res in A['node_res'].items():
        tc = A['node_tc'].get(h) or {}
        if not (tc.get('result') == 'pass' or tc.get('fail_phase') == 'performance'):
            continue
        n = run.nodes.get(h) or {}
        if res['stream'] and res['stream_valid']:
            for m in ('triad', 'copy'):
                if res['stream'].get(m) is not None:
                    cpu.setdefault(run.cpu_group(h), {}).setdefault(m, []).append(res['stream'][m])
        devs = [res['gpus'][u] for u in res['gpu_order']]
        if devs and n.get('gpu_model'):
            for m in GPU_METRICS:
                vals = [d[m] for d in devs if d.get(m) is not None]
                if vals:
                    gpu.setdefault('gpu:' + n['gpu_model'], {}).setdefault(m + '_min', []).append(min(vals))
    refs = {}
    for d in (cpu, gpu):
        for g in sorted(d, key=natkey):
            refs[g] = dict((m, round(median(v), 3)) for m, v in sorted(d[g].items()))
    out = {'schema': 1, 'source_run': run.run_id, 'SolNodeCheck': refs}
    tmp = path + '.tmp'
    with open(tmp, 'w') as fp:
        json.dump(out, fp, indent=1)
        fp.write('\n')
    os.replace(tmp, path)
    return refs


def terminal(run, A, P, issues, counts, files, rerun, baseline_id, mpi_note=''):
    out = []
    W = WIDTH

    def wrap(text, indent, first=''):
        return textwrap.wrap(text, W, initial_indent=first, subsequent_indent=' ' * indent,
                             break_on_hyphens=False) or [first.rstrip()]

    head = 'Sol pre-release check  run=%s  reservation=%s' % (run.run_id, run.inv.get('reservation') or
                                                              sol.env('RESERVATION', '-') or '-')
    if baseline_id:
        head += '  baseline=%s' % baseline_id
    if A.get('parent'):
        head += '  parent=%s' % A['parent']['id']
    out.append(head)
    if counts['not_tested'] is None:
        line = 'Nodes: %d tested | %d passed | %d with problems | not tested and excluded unknown (no nodes.json)' % (
            counts['testable'], counts['passed'], counts['problems'])
    else:
        line = 'Nodes: %d testable | %d passed | %d with problems | %d not tested | %d excluded' % (
            counts['testable'], counts['passed'], counts['problems'], counts['not_tested'], counts['excluded'])
    if counts['not_selected']:
        line += ' | %d not selected (--only)' % counts['not_selected']
    out.append(line)
    for w in run.warnings:
        out.extend(wrap(w, 9, 'WARNING: '))
    if not run.have_node_stage:
        out.append('NOT VERIFIED: no node-stage results (stage_node.json missing or unreadable), so the exit code is 1')
    if issues:
        out += ['', 'SYSTEMIC (shared by many nodes; the per-node rows are in problems.csv)']
        for s in issues:
            unit = 'links among %d nodes' % len(s['hosts']) if s['category'] == 'SLOW_LINK' else 'nodes'
            text = '[%s] %s on %d %s %s: %s' % (s['check'], s['category'], s['count'], unit,
                                                short_hostlist(s['hosts'], 72), s['reason'])
            out.extend(wrap(text, 4, '  '))
    rows = [p for p in P if p['node'] and not p.get('systemic')]
    out += ['', 'PROBLEM NODES (worst first)']
    if not rows:
        out.append('  none')
    else:
        wn = max(8, max(len(p['node']) for p in rows)) + 2
        wc = max(len(p['category']) for p in rows) + 2
        wk = max(9, max(len(p['check']) for p in rows)) + 2
        out.append('  ' + 'NODE'.ljust(wn) + 'CATEGORY'.ljust(wc) + 'CHECK'.ljust(wk) + 'REASON')
        for p in rows:
            first = '  ' + p['node'].ljust(wn) + p['category'].ljust(wc) + p['check'].ljust(wk)
            out.extend(wrap(p['reason'], len(first), first))
    if run.stage == 'all':
        out += ['', 'MPI JOBS']
        mrows = terminal_mpi_rows(A['mpi_rows'])
        if not run.have_mpi_stage:
            out.append('  not run (no stage_mpi.json)')
        elif not mrows:
            out.append('  no MPI test cases in stage_mpi.json')
        else:
            wg = max(8, max(len(r['group']) for r in mrows)) + 2
            wm = max(8, max(len(r['mpi']) for r in mrows)) + 1
            wt = max(20, max(len(r['test']) for r in mrows)) + 1
            out.append('  ' + 'GROUP'.ljust(wg) + 'MPI'.ljust(wm) + 'TEST'.ljust(wt) + 'RESULT  KEY NUMBERS')
            prev = None
            for r in mrows:
                if prev is not None and r['group'] != prev:
                    out.append('')
                prev = r['group']
                res = {'pass': 'PASS', 'fail': 'FAIL', 'fail_deps': 'SKIP', 'skip': 'SKIP',
                       'not run': 'NOTRUN'}.get(r['result'], str(r['result']).upper())
                first = '  ' + r['group'].ljust(wg) + r['mpi'].ljust(wm) + r['test'].ljust(wt) + res.ljust(8)
                out.extend(wrap(mpi_numbers(r, W - len(first)), len(first), first))
    out.append('')
    if mpi_note:
        out.extend(wrap(mpi_note, 2))
    line = 'Files: %s  %s  %s' % files
    if len(line) > W:
        line = 'Files in %s:\n  %s  %s  %s' % ((os.path.dirname(files[0]) or '.',) + (os.path.basename(files[0]),)
                                                + files[1:])
    out.append(line)
    if rerun:
        out.extend(rerun_command(rerun, 'Rerun problem nodes: '))
    return '\n'.join(out)


def short_hostlist(names, limit):
    '''Compressed hostlist, cut at a top-level comma with ",..." when longer than limit.'''
    full = hostlist(names)
    if len(full) <= limit:
        return full
    out, depth = '', 0
    for i, ch in enumerate(full):
        depth += (ch == '[') - (ch == ']')
        if ch == ',' and depth == 0:
            if i > limit:
                break
            out = full[:i]
    return (out or full[:limit]) + ',...'


def rerun_command(nodes, label, width=WIDTH):
    '''The --only command on one line when it fits. Otherwise the label goes on its own line and the
    hostlist is split into chunks joined as 'a,b,'\\ + newline + 'c,d', so a pasted copy still gives bash one
    argument; every chunk is itself a valid hostlist.'''
    full = hostlist(nodes)
    cmd = "./run_prerelease.sh --only '%s'" % full
    if len(label) + len(cmd) <= width:
        return [label + cmd]
    nodes = sorted(set(nodes), key=natkey)
    lines, chunk, first = [], [], True
    for h in nodes:
        lead = "./run_prerelease.sh --only '" if first else "'"
        if chunk and len(lead) + len(hostlist(chunk + [h])) + 3 > width:
            lines.append(lead + hostlist(chunk) + ",'\\")
            chunk, first = [], False
        chunk.append(h)
    lines.append(("./run_prerelease.sh --only '" if first else "'") + hostlist(chunk) + "'")
    return [label.rstrip()] + lines


def terminal_mpi_rows(rows):
    '''MPI rows in display order; tests skipped because the same dependency failed collapse into one row.'''
    out, seen = [], {}
    for r in sorted(rows, key=mpi_row_key):
        if r['result'] == 'fail_deps':
            key = (r['group'], r['mpi'], r['reason'])
            if key in seen:
                seen[key]['count'] += 1
                seen[key]['test'] = 'osu (%d tests)' % seen[key]['count']
                continue
            r = dict(r, count=1)
            seen[key] = r
        out.append(r)
    return sorted(out, key=mpi_row_key)


def mpi_row_key(r):
    groups = [g['name'] for g in sol.mpi_groups()]
    gi = groups.index(r['group']) if r['group'] in groups else (-1 if r['group'] == '-' else len(groups))
    ti = TEST_ORDER.index(r['test']) if r['test'] in TEST_ORDER else len(TEST_ORDER)
    mi = MPI_ORDER.index(r['mpi']) if r['mpi'] in MPI_ORDER else -1
    return (gi, r['group'], ti, r['test'], mi, r['mpi'])


def mpi_numbers(r, width=None):
    '''"lat_8B 12.3 us  lat_1M 410 us"; a unit shared with the next value is dropped only if the line
    would not fit otherwise.'''
    if r['result'] != 'pass':
        return r['reason'] or str(r['result'])
    head = []
    if r['test'] == 'pairwise' and r['nodes']:
        head.append('n=%d' % len(r['nodes']))
    elif r['test'].startswith('collective/') and r['nodes'] and r.get('tpn'):
        head.append('%dx%d ranks' % (len(r['nodes']), r['tpn']))
    perf = r['perf']
    for compact in (False, True):
        parts = list(head)
        for i, (var, v, unit) in enumerate(perf):
            same_next = compact and i + 1 < len(perf) and perf[i + 1][2] == unit
            parts.append('%s %s' % (var, fmt(v, '' if same_next else unit)))
        text = '  '.join(parts) or 'no performance values'
        if r.get('nodata'):
            text += '  (no output file: no curve)'
        if width is None or len(text) <= width:
            break
    return text


# --------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description='Sol pre-release report')
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--baseline')
    ap.add_argument('--parent', help='full run whose medians and driver majority a small --only rerun is '
                                     'compared with')
    ap.add_argument('--stage', choices=['node', 'all'], default='all')
    ap.add_argument('--failed-nodes-out')
    ap.add_argument('--write-references')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)
    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print('report.py: run directory not found: %s' % args.run_dir, file=sys.stderr)
        return 2
    run = Run(run_dir, args.stage)
    apply_settings(run.inv)
    of = sol.env_float('OUTLIER_FRACTION', 0.90)
    hd = sol.env_float('HISTORY_DROP', 0.10)
    refs_path = sol.env('REFERENCES')
    if refs_path and os.path.isfile(refs_path):
        try:
            with open(refs_path) as fp:
                REFERENCE_SOURCE[0] = 'run %s' % json.load(fp).get('source_run', '?')
        except (OSError, ValueError, AttributeError):
            pass
    parent = None
    if args.parent and not args.write_references:
        pdir = os.path.abspath(args.parent)
        if not os.path.isdir(pdir):
            run.warnings.append('parent run directory not found: %s' % args.parent)
        elif os.path.realpath(pdir) == os.path.realpath(run_dir):
            run.warnings.append('parent is this run; ignored')
        else:
            try:
                parent = load_parent(pdir, of)
            except (OSError, ValueError, csv.Error) as e:
                run.warnings.append('parent run %s unreadable: %s' % (args.parent, e))
            if parent is not None and not parent['medians'] and not parent['versions']:
                run.warnings.append('parent run %s has no node-stage values to compare with' % parent['id'])
                parent = None
    A = analyze(run, of, parent)

    if args.write_references:
        refs = write_references(args.write_references, run, A)
        if not args.quiet:
            print('wrote %s: %d groups from run %s' % (args.write_references, len(refs), run.run_id))
        return 0 if refs else 1

    P = A['problems']
    baseline_id, base_chart = None, {}
    if args.baseline:
        bdir = os.path.abspath(args.baseline)
        if not os.path.isdir(bdir):
            run.warnings.append('baseline directory not found: %s' % args.baseline)
        elif os.path.realpath(bdir) == os.path.realpath(run_dir):
            run.warnings.append('baseline is this run; no comparison')
        else:
            base, baseline_id = baseline_metrics(bdir, of)
            if not base:
                run.warnings.append('baseline %s has no metrics' % args.baseline)
            rows, base_chart = baseline_compare(A, base, baseline_id, hd)
            P.extend(rows)
    P = merge_perf_rows(P)
    issues = find_systemic(P, run)
    P = order_rows(P)
    node_rows = [p for p in P if p['node'] and '+' not in p['node']]

    inv = bool(run.nodes)
    status = dict((h, node_status(n)) for h, n in run.nodes.items())
    states = list(status.values())
    if inv:
        testable = [h for h, s in status.items() if s == 'testable']
    else:            # no nodes.json: the node-stage test cases are all that is known
        testable = sorted(A['node_tc'], key=natkey)
    with_problems = set(p['node'] for p in node_rows if p['severity'] <= 8 and p['node'] in set(testable))
    counts = {
        'testable': len(testable), 'problems': len(with_problems),
        # without node-stage results nothing was verified, so no node passed
        'passed': len([h for h in testable if h not in with_problems]) if run.have_node_stage else 0,
        'unavailable': states.count('unavailable'), 'untestable': states.count('untestable'),
        'not_tested': states.count('unavailable') + states.count('untestable') if inv else None,
        'excluded': states.count('excluded') if inv else None,
        'not_selected': states.count('not_selected'),
    }
    rerun = sorted(set(p['node'] for p in node_rows if p['severity'] <= SEVERITY['OUTLIER']), key=natkey)

    # the wrapper needs the failed-nodes list even if a later output fails
    if args.failed_nodes_out:
        failed = sorted(set(p['node'] for p in node_rows if p['category'] in ('JOB_FAILED', 'WRONG_RESULT')),
                        key=natkey)
        tmp = args.failed_nodes_out + '.tmp'
        with open(tmp, 'w') as fp:
            fp.write(''.join(h + '\n' for h in failed))
        os.replace(tmp, args.failed_nodes_out)

    write_csv(os.path.join(run_dir, 'problems.csv'),
              ['severity', 'category', 'node', 'group', 'check', 'metric', 'value', 'expected', 'reason'],
              [[p['severity'], p['category'], p['node'], p['group'], p['check'], p['metric'], csv_num(p['value']),
                csv_num(p['expected']), p['reason']] for p in P])
    write_csv(os.path.join(run_dir, 'metrics.csv'),
              ['run_id', 'node', 'group', 'device', 'device_name', 'check', 'metric', 'value', 'unit'],
              [[run.run_id, m[0], m[1], m[2], m[3], m[4], m[5], csv_num(m[6]), m[7]]
               for m in sorted(A['metrics'], key=lambda m: (m[4], natkey(m[1]), natkey(m[0]), m[2], natkey(m[5])))])
    write_csv(os.path.join(run_dir, 'versions.csv'), VERSIONS_HEADER, version_rows(run, A))

    bad = [p for p in node_rows if p['severity'] <= EXIT_SEVERITY]
    mpi_bad = mpi_failures(A) if run.stage == 'all' else []
    mpi_text = mpi_failure_text(mpi_bad)
    # the terminal says why the exit code is 1 when only the MPI stage caused it
    mpi_note = ('Exit code 1 because of the MPI stage only: %s (see MPI JOBS). No node has a problem from '
                'UNAVAILABLE to TOO_SLOW.' % mpi_text) if mpi_bad and not bad and run.have_node_stage else ''
    html_path = os.path.join(run_dir, 'report.html')
    try:
        write_html(html_path, run, A, P, issues, counts, rerun, baseline_id, base_chart, of, hd, mpi_text=mpi_text)
    except Exception as e:    # the text outputs matter more than the page
        run.warnings.append('report.html not written: %s: %s' % (type(e).__name__, e))
        print('report.py: report.html not written: %s: %s' % (type(e).__name__, e), file=sys.stderr)

    if not args.quiet:
        rel = os.path.relpath(html_path)
        shown = rel if not rel.startswith('..') else html_path
        print(terminal(run, A, P, issues, counts, (shown, 'problems.csv', 'metrics.csv'), rerun, baseline_id,
                       mpi_note=mpi_note))
    # no node-stage results means nothing was verified: never report success then
    return 1 if bad or mpi_bad or not run.have_node_stage else 0


def mpi_failures(A):
    '''MPI-stage rows that make the exit code 1: failed or aborted tests, failed OSU builds and the tests they
    left unrun. A group left out of the stage for too few usable nodes (NOTRUN) does not count.'''
    return [r for r in A['mpi_rows'] if r['result'] not in ('pass', 'not run')]


def mpi_failure_text(rows):
    '''"2 tests failed; the OSU build for mpich failed, so 16 tests did not run" for mpi_failures() rows.'''
    if not rows:
        return ''
    builds = sorted(set(r['mpi'] for r in rows if r['test'] == 'build'))
    skipped = sum(1 for r in rows if r['test'] != 'build' and r['result'] in ('fail_deps', 'skip'))
    failed = sum(1 for r in rows if r['test'] != 'build' and r['result'] not in ('fail_deps', 'skip'))
    parts = []
    if failed:
        parts.append('%d test%s failed' % (failed, '' if failed == 1 else 's'))
    not_run = '%d test%s did not run' % (skipped, '' if skipped == 1 else 's')
    if builds:
        parts.append('the OSU build%s for %s failed%s' % ('' if len(builds) == 1 else 's', ', '.join(builds),
                                                          ', so ' + not_run if skipped else ''))
    elif skipped:
        parts.append(not_run)
    return '; '.join(parts)


# --------------------------------------------------------------------------------------------
# report.html: one self-contained file. Tables are rendered here (escaped); the charts are drawn by the
# inline script from the embedded JSON, at the width of their container.

STATUS = {1: 'critical', 2: 'critical', 3: 'critical', 4: 'serious', 5: 'serious', 6: 'warning', 7: 'warning',
          8: 'warning', 9: 'info'}
SERIES_SLOT = {'openmpi': 1, 'mpich': 2, 'hpcx': 3, 'hpcx-cuda': 3}     # GPU-direct runs hpcx with CUDA

CSS = r'''
.viz-root{color-scheme:light;
 --surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
 --border:rgba(11,11,11,.10);--shadow:rgba(11,11,11,.10);
 --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;
 --good:#0ca30c;--warning:#fab219;--serious:#ec835a;--critical:#d03b3b;
 --div-pos:#2a78d6;--div-neg:#e34948;--div-mid:#f0efec;--zone:rgba(208,59,59,.055);
 --q0:#cde2fb;--q1:#b7d3f6;--q2:#9ec5f4;--q3:#86b6ef;--q4:#6da7ec;--q5:#5598e7;--q6:#3987e5;--q7:#2a78d6;
 --q8:#256abf;--q9:#1c5cab;--q10:#184f95;--q11:#104281;--q12:#0d366b;--qn:#e1e0d9}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
 --surface:#1a1a19;--page:#0d0d0d;--ink:#f0efec;--ink-2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;
 --border:rgba(255,255,255,.10);--shadow:rgba(0,0,0,.5);
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--div-mid:#383835;--zone:rgba(208,59,59,.10);
 --q0:#0d366b;--q1:#104281;--q2:#184f95;--q3:#1c5cab;--q4:#256abf;--q5:#2a78d6;--q6:#3987e5;--q7:#5598e7;
 --q8:#6da7ec;--q9:#86b6ef;--q10:#9ec5f4;--q11:#b7d3f6;--q12:#cde2fb;--qn:#2c2c2a}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;
 --surface:#1a1a19;--page:#0d0d0d;--ink:#f0efec;--ink-2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;
 --border:rgba(255,255,255,.10);--shadow:rgba(0,0,0,.5);
 --s1:#3987e5;--s2:#d95926;--s3:#199e70;--div-mid:#383835;--zone:rgba(208,59,59,.10);
 --q0:#0d366b;--q1:#104281;--q2:#184f95;--q3:#1c5cab;--q4:#256abf;--q5:#2a78d6;--q6:#3987e5;--q7:#5598e7;
 --q8:#6da7ec;--q9:#86b6ef;--q10:#9ec5f4;--q11:#b7d3f6;--q12:#cde2fb;--qn:#2c2c2a}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1200px;margin:0 auto;padding:24px 16px 40px}
h1{font-size:24px;line-height:1.2;margin:0 0 6px;font-weight:650}
h2{font-size:19px;margin:36px 0 6px;font-weight:650}
h3{font-size:15px;margin:0;font-weight:600}
p{margin:0 0 8px}
a{color:inherit}
.meta{color:var(--ink-2);font-size:13px;margin:0 0 2px;overflow-wrap:anywhere}
.lead{color:var(--ink-2);margin:0 0 14px;max-width:80ch}
nav.toc{display:flex;flex-wrap:wrap;gap:4px 16px;margin:14px 0 0;font-size:13px}
nav.toc a{color:var(--ink-2);text-decoration:none;border-bottom:1px solid var(--axis)}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px;margin:0 0 16px;min-width:0}
.verdict{display:flex;gap:10px;align-items:flex-start;font-size:15px;margin:20px 0 12px}
.verdict .ico{width:20px;height:20px;margin-top:1px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:0 0 8px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.tile .l{color:var(--ink-2);font-size:13px}
.tile .v{font-size:30px;font-weight:600;line-height:1.2}
.tile .d{color:var(--muted);font-size:12px}
.ico{width:16px;height:16px;flex:none;vertical-align:-3px}
.badge{white-space:nowrap;font-weight:600;font-size:12px}
.badge .ico{margin-right:6px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 14px 6px 0;border-bottom:1px solid var(--grid);vertical-align:top}
.ptable td{vertical-align:baseline}
th{color:var(--ink-2);font-weight:600;font-size:12px;position:sticky;top:0;background:var(--surface)}
td.num,th.num{text-align:right}
td.reason{min-width:16em;overflow-wrap:anywhere}
td.mono,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px}
.tscroll{overflow:auto}
.tablewrap{overflow:auto;max-height:540px;margin-top:8px}
.fig-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start}
.fig-head .sub{color:var(--ink-2);font-size:13px;margin:2px 0 0;max-width:90ch}
button.tbl-toggle{font:inherit;font-size:12px;color:var(--ink);background:var(--page);border:1px solid var(--axis);
 border-radius:6px;padding:4px 10px;cursor:pointer;flex:none}
button.tbl-toggle:hover{border-color:var(--muted)}
button.tbl-toggle:focus-visible,.plot svg:focus-visible,.plot [tabindex]:focus-visible{outline:2px solid var(--s1);outline-offset:2px}
.key{display:flex;flex-wrap:wrap;gap:4px 16px;color:var(--ink-2);font-size:12px;margin:10px 0 2px;align-items:center}
.key span{display:inline-flex;align-items:center;gap:6px}
.key svg{flex:none}
.plot{width:100%;margin-top:6px;min-height:20px}
.plot svg{display:block;overflow:visible}
.heatscroll{overflow-x:auto;overflow-y:hidden;margin-top:6px}
.heatscroll .plot{width:auto}
svg text{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;fill:var(--ink-2)}
.grid line{stroke:var(--grid);stroke-width:1;shape-rendering:crispEdges}
.axis{stroke:var(--axis);stroke-width:1;shape-rendering:crispEdges}
.tick text{font-variant-numeric:tabular-nums;fill:var(--muted);font-size:11px}
text.unit{fill:var(--muted);font-size:11px}
text.rl{fill:var(--ink);font-size:12px;font-weight:600}
tspan.rm{fill:var(--muted);font-weight:400}
text.fl{fill:var(--ink);font-size:11px;font-weight:600}
rect.zone{fill:var(--zone)}
line.zedge{stroke:var(--critical);stroke-width:1;stroke-opacity:.55;shape-rendering:crispEdges}
line.med{stroke:var(--ink);stroke-width:2;stroke-linecap:round}
circle.dot{fill:var(--s1);stroke:var(--surface);stroke-width:2}
circle.dot.flag{fill:var(--critical)}
circle.dneg{fill:var(--div-neg);stroke:var(--surface);stroke-width:2}
circle.dpos{fill:var(--div-pos);stroke:var(--surface);stroke-width:2}
circle.dmid{fill:var(--muted);stroke:var(--surface);stroke-width:2}
rect.band{fill:var(--div-mid)}
line.zero{stroke:var(--axis);stroke-width:1.5}
circle.hl{fill:none;stroke:var(--ink);stroke-width:2;pointer-events:none}
rect.hlc{fill:none;stroke:var(--ink);stroke-width:2;pointer-events:none}
.cells rect{shape-rendering:crispEdges}
.cells rect:focus{outline:none}
text.hm{font-size:10px;fill:var(--ink-2)}
text.hm.f{fill:var(--ink);font-weight:700}
circle.hmf{fill:var(--critical)}
.q0{fill:var(--q0)}.q1{fill:var(--q1)}.q2{fill:var(--q2)}.q3{fill:var(--q3)}.q4{fill:var(--q4)}.q5{fill:var(--q5)}
.q6{fill:var(--q6)}.q7{fill:var(--q7)}.q8{fill:var(--q8)}.q9{fill:var(--q9)}.q10{fill:var(--q10)}.q11{fill:var(--q11)}
.q12{fill:var(--q12)}.qn{fill:var(--qn)}
.ramp{display:flex;flex-wrap:wrap;align-items:flex-start;gap:14px;margin-top:10px;font-size:11px;color:var(--ink-2)}
.ramp .bar{display:grid;grid-template-columns:repeat(13,18px);row-gap:3px}
.ramp .bar i{height:12px;display:block}
.ramp .bar b{font-weight:400;grid-row:2;white-space:nowrap;font-variant-numeric:tabular-nums}
.ramp .nd{display:inline-flex;gap:6px;align-items:center}
.ramp .nd i{width:18px;height:12px;display:inline-block;background:var(--qn)}
path.ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.c1{stroke:var(--s1)}.c2{stroke:var(--s2)}.c3{stroke:var(--s3)}.c0{stroke:var(--muted)}
circle.ep{stroke:var(--surface);stroke-width:2}
circle.f1{fill:var(--s1)}circle.f2{fill:var(--s2)}circle.f3{fill:var(--s3)}circle.f0{fill:var(--muted)}
line.lead{stroke:var(--axis);stroke-width:1}
text.el{font-size:11px;fill:var(--ink)}
line.xh{stroke:var(--muted);stroke-width:1;pointer-events:none}
.lkey{display:inline-block;width:14px;height:2px;border-radius:1px;vertical-align:middle}
.k1{background:var(--s1)}.k2{background:var(--s2)}.k3{background:var(--s3)}.k0{background:var(--muted)}
.osu-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:16px;margin-top:12px}
.osu-grid figure{margin:0;min-width:0}
figure{margin:0}
.mpi-table{table-layout:fixed}
.mpi-table td,.mpi-table th{padding-right:14px;overflow-wrap:anywhere}
.mpi-table th:first-child{width:19%}
.res{display:inline-flex;gap:6px;align-items:center;font-weight:600;font-size:12px}
.small{font-size:12px;color:var(--ink-2)}
details{margin:6px 0 0}
details summary{cursor:pointer;color:var(--ink-2);font-size:13px}
.sys{display:flex;gap:10px;align-items:flex-start;padding:10px 0;border-bottom:1px solid var(--grid)}
.sys:last-child{border-bottom:0}
.sys .t{font-weight:600}
.hosts{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;overflow-wrap:anywhere}
pre.cmd{background:var(--page);border:1px solid var(--border);border-radius:8px;padding:10px 12px;white-space:pre-wrap;
 overflow-wrap:anywhere;font-size:12px;margin:6px 0 0}
#tip{position:fixed;z-index:20;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--border);
 box-shadow:0 6px 24px var(--shadow);border-radius:8px;padding:8px 10px;font-size:12px;max-width:min(340px,90vw)}
#tip[hidden]{display:none}
#tip .tv{font-size:13px}
#tip .tl{color:var(--ink-2)}
#tip .tk{display:inline-block;width:12px;height:2px;margin-right:6px;vertical-align:middle;border-radius:1px}
footer{color:var(--ink-2);font-size:13px;margin-top:32px}
@media (max-width:640px){h1{font-size:21px}.tile .v{font-size:26px}.card{padding:12px}th,td{padding-right:8px}
 td.reason{min-width:0}
 .ptable thead{display:none}
 .ptable tr{display:block;padding:8px 0;border-bottom:1px solid var(--grid)}
 .ptable td{display:inline-block;border:0;padding:0 12px 2px 0;vertical-align:baseline}
 .ptable td.reason{display:block;padding:3px 0 0}
 .mpi-table thead{display:none}
 .mpi-table tr{display:block;padding:8px 0;border-bottom:1px solid var(--grid)}
 .mpi-table td{display:block;border:0;padding:2px 0}
 .mpi-table td:first-child{font-weight:600;padding-bottom:4px}
 .mpi-table td[data-label]::before{content:attr(data-label);display:block;color:var(--ink-2);font-size:12px}
 .osu-grid{gap:12px}}
'''

JS = r'''
(function () {
  'use strict';
  var D = JSON.parse(document.getElementById('report-data').textContent);
  var NS = 'http://www.w3.org/2000/svg';
  var tip = document.getElementById('tip');

  function S(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    if (attrs) {
      for (var k in attrs) {
        var v = attrs[k];
        if (v === null || v === undefined) { continue; }
        e.setAttribute(k, typeof v === 'number' ? Math.round(v * 10) / 10 : v);   // SVG numbers: 1 decimal
      }
    }
    if (parent) { parent.appendChild(e); }
    return e;
  }
  function T(tag, attrs, text, parent) { var e = S(tag, attrs, parent); e.textContent = text; return e; }
  function H(tag, cls, text, parent) {
    var e = document.createElement(tag);
    if (cls) { e.className = cls; }
    if (text !== null && text !== undefined) { e.textContent = text; }
    if (parent) { parent.appendChild(e); }
    return e;
  }
  function r1(v) { return Math.round(v * 10) / 10; }
  function fmt(v, unit) {
    if (v === null || v === undefined || isNaN(v)) { return 'n/a'; }
    var a = Math.abs(v), s;
    if (a >= 1000) { s = Math.round(v).toLocaleString('en-US'); }
    else if (a >= 100) { s = v.toFixed(0); }
    else if (a >= 10) { s = v.toFixed(1); }
    else if (a >= 1) { s = v.toFixed(2); }
    else { s = String(+v.toPrecision(3)); }
    return unit ? s + ' ' + unit : s;
  }
  function trimNum(x) { return String(Math.round(x * 10) / 10); }
  function compact(v) {
    var a = Math.abs(v);
    if (a >= 1e9) { return trimNum(v / 1e9) + 'G'; }
    if (a >= 1e6) { return trimNum(v / 1e6) + 'M'; }
    if (a >= 1e3) { return trimNum(v / 1e3) + 'K'; }
    if (a > 0 && a < 1) { return String(+v.toPrecision(2)); }
    return trimNum(v);
  }
  function bytes(v) {
    if (v >= 1048576) { return (v / 1048576) + 'M'; }
    if (v >= 1024) { return (v / 1024) + 'K'; }
    return String(v);
  }
  function lin(d0, d1, r0, r1_) { return function (v) { return r0 + (v - d0) / (d1 - d0) * (r1_ - r0); }; }
  function logsc(d0, d1, r0, r1_) {
    var l0 = Math.log(d0), l1 = Math.log(d1);
    return function (v) { return r0 + (Math.log(v) - l0) / (l1 - l0) * (r1_ - r0); };
  }
  function niceTicks(lo, hi, n) {
    if (!isFinite(lo) || !isFinite(hi) || !(hi > lo)) { return [isFinite(lo) ? lo : 0]; }
    var span = hi - lo, step = Math.pow(10, Math.floor(Math.log(span / n) / Math.LN10)), err = span / n / step;
    if (err >= 7.5) { step *= 10; } else if (err >= 3.5) { step *= 5; } else if (err >= 1.5) { step *= 2; }
    var t = [];
    for (var v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) { t.push(+v.toPrecision(12)); }
    return t;
  }
  // log-axis ticks: 1-2-5 per decade, else 1-3, else decades (thinned), whichever first fits maxN
  function logTicks(lo, hi, maxN) {
    if (!(lo > 0) || !isFinite(lo) || !isFinite(hi) || !(hi > lo)) { return []; }
    var sets = [[1, 2, 5], [1, 3], [1]], e0 = Math.floor(Math.log(lo) / Math.LN10), e1 = Math.ceil(Math.log(hi) / Math.LN10);
    for (var k = 0; k < sets.length; k++) {
      var t = [];
      for (var e = e0; e <= e1; e++) {
        sets[k].forEach(function (m) { var v = +(m * Math.pow(10, e)).toPrecision(6); if (v >= lo * 0.999 && v <= hi * 1.001) { t.push(v); } });
      }
      if (t.length <= maxN || k === sets.length - 1) {
        while (t.length > maxN) { t = t.filter(function (v, i) { return i % 2 === 0; }); }
        return t;
      }
    }
    return [];
  }
  function measure(svg, text, cls) {
    var t = T('text', {x: -999, y: -999, 'class': cls}, text, svg);
    var w = t.getComputedTextLength();
    svg.removeChild(t);
    return w;
  }

  // tooltip: rows = [{v: value, l: label, key: series slot}], value first and strong
  function showTip(rows, x, y) {
    tip.textContent = '';
    rows.forEach(function (r, i) {
      var line = H('div', i === 0 ? 'tv' : 'tl', null, tip);
      if (r.key !== undefined) { H('span', 'tk k' + r.key, null, line); }
      if (r.v !== undefined) { H('b', null, r.v, line); }
      if (r.l !== undefined) { H('span', null, (r.v !== undefined ? '  ' : '') + r.l, line); }
    });
    tip.hidden = false;
    var tw = tip.offsetWidth, th = tip.offsetHeight, vw = window.innerWidth, vh = window.innerHeight;
    var left = x + 14, top = y + 14;
    if (left + tw > vw - 8) { left = x - tw - 14; }
    if (left < 8) { left = 8; }
    if (top + th > vh - 8) { top = y - th - 14; }
    if (top < 8) { top = 8; }
    tip.style.left = left + 'px';
    tip.style.top = top + 'px';
  }
  function hideTip() { tip.hidden = true; }
  function tipAt(el, rows) { var b = el.getBoundingClientRect(); showTip(rows, b.right, b.bottom); }
  function svgPoint(svg, e) { var b = svg.getBoundingClientRect(); return {x: e.clientX - b.left, y: e.clientY - b.top}; }

  // vertical placement of dots in a row: beeswarm inside the band, overlapping only when the band is full
  function swarm(xs, cy, maxOff, r) {
    var placed = [], ys = [], step = r * 1.15, k, n = Math.floor(maxOff / step);
    xs.forEach(function (px) {
      var best = 0, bestOv = 1e9;
      for (k = 0; k <= 2 * n; k++) {
        var off = (k % 2 ? 1 : -1) * Math.ceil(k / 2) * step, ov = 0;
        for (var q = 0; q < placed.length; q++) {
          var dx = placed[q][0] - px, dy = placed[q][1] - (cy + off);
          if (dx * dx + dy * dy < 4 * r * r) { ov++; }
        }
        if (ov === 0) { best = off; break; }
        if (ov < bestOv) { bestOv = ov; best = off; }
      }
      placed.push([px, cy + best]);
      ys.push(cy + best);
    });
    return ys;
  }

  // direct labels for flagged dots: left or right of the dot, avoiding each other and the plot edges
  function placeLabels(svg, items, left, right, preferRight) {
    var boxes = [];
    items.forEach(function (it) {
      var w = measure(svg, it.text, 'fl'), h = 12;
      var cands = preferRight ? ['r', 'l', 'ru', 'lu', 'rd', 'ld'] : ['l', 'r', 'lu', 'ru', 'ld', 'rd'];
      var chosen = null;
      for (var c = 0; c < cands.length && !chosen; c++) {
        var side = cands[c][0], dy = cands[c][1] === 'u' ? -11 : (cands[c][1] === 'd' ? 11 : 0);
        var x0 = side === 'r' ? it.cx + 9 : it.cx - 9 - w, y0 = it.cy + dy - h / 2;
        if (x0 < left || x0 + w > right) { continue; }
        var hit = boxes.some(function (b) { return x0 < b[0] + b[2] + 3 && b[0] < x0 + w + 3 && y0 < b[1] + b[3] && b[1] < y0 + h; });
        if (!hit) { chosen = [x0, y0, w, h]; }
      }
      if (!chosen) { chosen = [Math.min(Math.max(left, it.cx + 9), right - w), it.cy - 6, w, h]; }
      boxes.push(chosen);
      T('text', {x: chosen[0], y: chosen[1] + 9.5, 'class': 'fl'}, it.text, svg);
    });
  }

  // keyboard + pointer behaviour shared by the dot charts
  function interact(svg, marks, rowsFor) {
    if (!marks.length) { return; }
    var ring = S('circle', {r: 8.5, 'class': 'hl', visibility: 'hidden'}, svg);
    var active = marks[0];
    active.el.setAttribute('tabindex', '0');
    marks.forEach(function (mk, i) { mk.i = i; });
    function show(mk) { ring.setAttribute('cx', mk.cx); ring.setAttribute('cy', mk.cy); ring.setAttribute('visibility', 'visible'); }
    function clear() { ring.setAttribute('visibility', 'hidden'); hideTip(); }
    function setActive(mk) {
      if (active !== mk) { active.el.setAttribute('tabindex', '-1'); mk.el.setAttribute('tabindex', '0'); active = mk; }
    }
    svg.addEventListener('pointermove', function (e) {
      var pt = svgPoint(svg, e), best = null, bd = 14 * 14;
      for (var i = 0; i < marks.length; i++) {
        var dx = marks[i].cx - pt.x, dy = marks[i].cy - pt.y, d = dx * dx + dy * dy;
        if (d <= bd) { bd = d; best = marks[i]; }
      }
      if (best) { show(best); showTip(rowsFor(best), e.clientX, e.clientY); } else { clear(); }
    });
    svg.addEventListener('pointerleave', clear);
    marks.forEach(function (mk) {
      mk.el.addEventListener('focus', function () { setActive(mk); show(mk); tipAt(mk.el, rowsFor(mk)); });
      mk.el.addEventListener('blur', clear);
    });
    svg.addEventListener('keydown', function (e) {
      var cur = active, next = null, i;
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        var j = cur.i + (e.key === 'ArrowRight' ? 1 : -1);
        if (j >= 0 && j < marks.length && marks[j].row === cur.row) { next = marks[j]; }
      } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        var dir = e.key === 'ArrowDown' ? 1 : -1, row = cur.row + dir, cand = [];
        while (!cand.length && row >= 0 && row <= marks[marks.length - 1].row) {
          for (i = 0; i < marks.length; i++) { if (marks[i].row === row) { cand.push(marks[i]); } }
          row += dir;
        }
        cand.forEach(function (m) { if (!next || Math.abs(m.cx - cur.cx) < Math.abs(next.cx - cur.cx)) { next = m; } });
      } else if (e.key === 'Home') { next = marks[0]; }
      else if (e.key === 'End') { next = marks[marks.length - 1]; }
      else if (e.key === 'Escape') { clear(); return; }
      if (next) { e.preventDefault(); setActive(next); next.el.focus(); }
    });
  }

  // group name plus the longest of the meta texts that still fits in the width
  function rowLabel(svg, x, y, name, metas, width) {
    var t = T('text', {x: x, y: y, 'class': 'rl'}, name, svg);
    var avail = width - t.getComputedTextLength() - 8;
    for (var i = 0; i < metas.length; i++) {
      if (!metas[i]) { continue; }
      var s = S('tspan', {'class': 'rm', dx: 8}, t);
      s.textContent = metas[i];
      if (s.getComputedTextLength() <= avail) { return; }
      t.removeChild(s);
    }
  }

  // ---- performance by group: one row per group, one dot per node or device
  // height a row needs so swarm() can stack its densest dot-wide cluster (k dots: centre, then +-1, +-2 ...
  // dot spacings above and below), capped so a huge group overlaps a little instead of growing without end
  function bandFor(xs, rad) {
    var s = xs.slice().sort(function (a, b) { return a - b; }), k = 0, j = 0;
    for (var i = 0; i < s.length; i++) {
      while (s[i] - s[j] > 2 * rad) { j++; }
      k = Math.max(k, i - j + 1);
    }
    var off = Math.ceil((k - 1) / 2) * 2 * rad * 1.15;
    return Math.max(22, Math.min(120, Math.ceil(2 * (off + rad + 1))));
  }
  function dotR(n) { return n > 40 ? 4 : 5; }
  // draw order: dots far from the centre first, so overlapping clusters stay visually centred
  function drawOrder(marks, centre) {
    return marks.slice().sort(function (a, b) { return Math.abs(b.cx - centre) - Math.abs(a.cx - centre); });
  }
  function topAxis(svg, ticks, x, y, fmtTick) {
    var tg = S('g', {'class': 'tick'}, svg);
    ticks.forEach(function (t) { T('text', {x: x(t), y: y, 'text-anchor': 'middle'}, fmtTick(t), tg); });
  }

  // dots on a shared "% of the group median" axis (p = [value, node, device, flag, pct])
  function renderStrip(host, c) {
    var W = Math.max(280, host.clientWidth), narrow = W < 560;
    var m = {l: 4, r: 14, t: 22}, LH = 18, GAP = 10, AX = 40;
    var pcts = [];
    c.rows.forEach(function (r) { r.pts.forEach(function (p) { if (p[4] !== null) { pcts.push(p[4]); } }); });
    var lo = Math.min.apply(null, pcts.concat([c.lower ? 95 : c.cut - 5]));
    var hi = Math.max.apply(null, pcts.concat([c.lower ? c.cut + 5 : 105]));
    var span = hi - lo;
    lo -= span * 0.03;
    hi += span * 0.03;
    var ticks = niceTicks(lo, hi, narrow ? 4 : 8), x = lin(lo, hi, m.l, W - m.r);
    var rows = c.rows.map(function (r) {
      var pts = r.pts.filter(function (p) { return p[4] !== null; }).sort(function (a, b) { return a[4] - b[4]; });
      var rad = dotR(pts.length), xs = pts.map(function (p) { return x(p[4]); });
      return {r: r, pts: pts, xs: xs, rad: rad, bh: bandFor(xs, rad)};
    });
    var tops = [], yy = m.t;
    rows.forEach(function (L) { tops.push(yy); yy += LH + L.bh + GAP; });
    var H = yy + AX - GAP;
    var svg = S('svg', {width: W, height: H, viewBox: '0 0 ' + W + ' ' + H, role: 'group', 'aria-label': c.title}, host);
    var bottom = H - AX, gg = S('g', {'class': 'grid'}, svg);
    function pctLabel(t) { return r1(t) + '%'; }
    ticks.forEach(function (t) { S('line', {x1: x(t), x2: x(t), y1: m.t - 6, y2: bottom}, gg); });
    topAxis(svg, ticks, x, m.t - 10, pctLabel);
    var marks = [];
    rows.forEach(function (L, ri) {
      var r = L.r, y0 = tops[ri], BH = L.bh, by = y0 + LH, cy = by + BH / 2;
      var cnt = r.n + ' ' + c.what + (r.n === 1 ? '' : 's');
      var med = r.med === null ? '' : (r.n === 1 && !r.src ? ', ' : ', median ') + fmt(r.med, c.unit);
      // r.src: too few values in this run, so the dots are placed against the median of an earlier full run
      rowLabel(svg, m.l, y0 + 13, r.g, r.src ? [cnt + med + ' from ' + r.src + ' (' + r.srcn + ' ' + c.what + 's)',
        cnt + med + ' from ' + r.src, cnt + ', median from ' + r.src, cnt] :
        [cnt + med + (r.checked ? '' : ' (under 3: not checked)'),
        cnt + med + (r.checked ? '' : ', not checked'), cnt + med, cnt], W - m.l - m.r);
      if (r.checked) {
        var z0 = c.lower ? x(c.cut) : m.l, z1 = c.lower ? W - m.r : x(c.cut);
        S('rect', {x: z0, y: by, width: Math.max(0, z1 - z0), height: BH, 'class': 'zone'}, svg);
        S('line', {x1: x(c.cut), x2: x(c.cut), y1: by, y2: by + BH, 'class': 'zedge'}, svg);
      }
      if (r.med !== null && (r.n > 1 || r.src)) {
        S('line', {x1: x(100), x2: x(100), y1: by - 2, y2: by + BH + 4, 'class': 'med'}, svg);
      }
      var ys = swarm(L.xs, cy, Math.max(0, BH / 2 - L.rad - 1), L.rad);
      var rowMarks = L.pts.map(function (p, i) { return {p: p, row: ri, cx: L.xs[i], cy: ys[i], r: r}; });
      drawOrder(rowMarks.filter(function (mk) { return !mk.p[3]; }), x(100)).concat(rowMarks.filter(function (mk) { return mk.p[3]; })).forEach(function (mk) {
        mk.el = S('circle', {cx: mk.cx, cy: mk.cy, r: L.rad, 'class': mk.p[3] ? 'dot flag' : 'dot', tabindex: '-1',
          'aria-label': mk.p[1] + (mk.p[2] ? ' ' + mk.p[2] : '') + ': ' + fmt(mk.p[0], c.unit) + ', ' + Math.round(mk.p[4]) + '% of the group median' + (r.src ? ' from ' + r.src : '')}, svg);
      });
      marks = marks.concat(rowMarks);
      placeLabels(svg, rowMarks.filter(function (mk) { return mk.p[3]; }).map(function (mk) {
        return {cx: mk.cx, cy: mk.cy, text: mk.p[1]};
      }), m.l, W - m.r, c.lower);
    });
    S('line', {x1: m.l, x2: W - m.r, y1: bottom, y2: bottom, 'class': 'axis'}, svg);
    var tg = S('g', {'class': 'tick'}, svg);
    ticks.forEach(function (t) { T('text', {x: x(t), y: bottom + 15, 'text-anchor': 'middle'}, pctLabel(t), tg); });
    T('text', {x: W - m.r, y: bottom + 32, 'text-anchor': 'end', 'class': 'unit'},
      narrow ? '% of group median' : '% of the group median (each row’s median is in its label)', svg);
    interact(svg, marks, function (mk) {
      var p = mk.p, rows = [{v: fmt(p[0], c.unit)}, {l: p[1] + (p[2] ? '  ' + p[2] : '')}];
      rows.push({l: Math.round(p[4]) + '% of the ' + mk.r.g + ' median ' + fmt(mk.r.med, c.unit) +
        (mk.r.src ? ' from ' + mk.r.src : '')});
      if (p[3]) { rows.push({l: p[3]}); }
      return rows;
    });
  }

  // ---- change vs baseline: dots at % change, diverging colours, tolerance band (c.neg to c.pos) in the midpoint grey
  function renderBase(host, c) {
    var W = Math.max(280, host.clientWidth), narrow = W < 560;
    var m = {l: 4, r: 14, t: 22}, LH = 18, GAP = 10, AX = 40;
    var lo = 2 * c.neg, hi = Math.max(-2 * c.neg, c.pos);
    c.rows.forEach(function (r) { r.pts.forEach(function (p) { lo = Math.min(lo, p[0]); hi = Math.max(hi, p[0]); }); });
    var span = hi - lo;
    lo -= span * 0.03;
    hi += span * 0.03;
    var ticks = niceTicks(lo, hi, narrow ? 4 : 8), x = lin(lo, hi, m.l, W - m.r);
    var rows = c.rows.map(function (r) {
      var pts = r.pts.slice().sort(function (a, b) { return a[0] - b[0]; });
      var rad = dotR(pts.length), xs = pts.map(function (p) { return x(p[0]); });
      return {r: r, pts: pts, xs: xs, rad: rad, bh: bandFor(xs, rad)};
    });
    var tops = [], yy = m.t;
    rows.forEach(function (L) { tops.push(yy); yy += LH + L.bh + GAP; });
    var H = yy + AX - GAP;
    var svg = S('svg', {width: W, height: H, viewBox: '0 0 ' + W + ' ' + H, role: 'group', 'aria-label': c.title}, host);
    var bottom = H - AX, gg = S('g', {'class': 'grid'}, svg);
    ticks.forEach(function (t) { S('line', {x1: x(t), x2: x(t), y1: m.t - 6, y2: bottom}, gg); });
    topAxis(svg, ticks, x, m.t - 10, function (t) { return (t > 0 ? '+' : '') + r1(t) + '%'; });
    var marks = [];
    rows.forEach(function (L, ri) {
      var r = L.r, y0 = tops[ri], BH = L.bh, by = y0 + LH, cy = by + BH / 2;
      rowLabel(svg, m.l, y0 + 13, r.g, [r.pts.length + ' ' + c.what + (r.pts.length === 1 ? '' : 's')], W - m.l - m.r);
      S('rect', {x: x(c.neg), y: by, width: x(c.pos) - x(c.neg), height: BH, 'class': 'band'}, svg);
      S('line', {x1: x(0), x2: x(0), y1: by, y2: by + BH, 'class': 'zero'}, svg);
      var ys = swarm(L.xs, cy, Math.max(0, BH / 2 - L.rad - 1), L.rad);
      var rowMarks = L.pts.map(function (p, i) { return {p: p, row: ri, cx: L.xs[i], cy: ys[i], r: r}; });
      drawOrder(rowMarks, x(0)).forEach(function (mk) {
        var cls = mk.p[0] < c.neg ? 'dneg' : (mk.p[0] > c.pos ? 'dpos' : 'dmid');
        mk.el = S('circle', {cx: mk.cx, cy: mk.cy, r: L.rad, 'class': cls, tabindex: '-1',
          'aria-label': mk.p[1] + (mk.p[2] ? ' ' + mk.p[2] : '') + ': ' + (mk.p[0] > 0 ? '+' : '') + mk.p[0].toFixed(1) + '%'}, svg);
      });
      marks = marks.concat(rowMarks);
      placeLabels(svg, rowMarks.filter(function (mk) { return mk.p[0] < c.neg; }).map(function (mk) {
        return {cx: mk.cx, cy: mk.cy, text: mk.p[1]};
      }), m.l, W - m.r, false);
    });
    S('line', {x1: m.l, x2: W - m.r, y1: bottom, y2: bottom, 'class': 'axis'}, svg);
    var tg = S('g', {'class': 'tick'}, svg);
    ticks.forEach(function (t) { T('text', {x: x(t), y: bottom + 15, 'text-anchor': 'middle'}, (t > 0 ? '+' : '') + r1(t) + '%', tg); });
    T('text', {x: W - m.r, y: bottom + 31, 'text-anchor': 'end', 'class': 'unit'}, 'change vs baseline (negative = slower)', svg);
    interact(svg, marks, function (mk) {
      var p = mk.p;
      return [{v: (p[0] > 0 ? '+' : '') + p[0].toFixed(1) + '%'}, {l: p[1] + (p[2] ? '  ' + p[2] : '')},
              {l: 'now ' + fmt(p[3], c.unit) + ', baseline ' + fmt(p[4], c.unit)}];
    });
  }

  // ---- fabric heatmap: shortfall of each pair against the group median link bandwidth
  function renderHeat(host, hm) {
    var n = hm.nodes.length, lab = 46;
    var avail = host.parentNode.clientWidth;
    var cell = Math.max(7, Math.min(24, Math.floor((avail - lab - 10) / n)));
    var gap = cell >= 14 ? 2 : (cell >= 9 ? 1 : 0), size = n * cell, W = lab + size + 10, H = lab + size + 10;
    var svg = S('svg', {width: W, height: H, viewBox: '0 0 ' + W + ' ' + H, role: 'group', 'aria-label': 'pairwise bandwidth heatmap ' + hm.group}, host);
    var bw = new Float64Array(n * n), lt = new Float64Array(n * n), i, j;
    for (i = 0; i < n * n; i++) { bw[i] = NaN; lt[i] = NaN; }
    hm.pairs.forEach(function (q) { bw[q[0] * n + q[1]] = bw[q[1] * n + q[0]] = q[2]; lt[q[0] * n + q[1]] = lt[q[1] * n + q[0]] = q[3]; });
    var flag = {}; hm.flag.forEach(function (k) { flag[k] = true; });
    var every = Math.max(1, Math.ceil(12 / cell)), x0 = lab, y0 = lab;
    hm.nodes.forEach(function (name, k) {
      var show = flag[k] || (k % every === 0 && !(flag[k - 1] || flag[k + 1]) );
      if (!show) { return; }
      var cls = 'hm' + (flag[k] ? ' f' : '');
      T('text', {x: x0 - 5, y: y0 + k * cell + cell / 2 + 3.5, 'text-anchor': 'end', 'class': cls}, name, svg);
      T('text', {x: 0, y: 0, 'text-anchor': 'start', 'class': cls, transform: 'translate(' + (x0 + k * cell + cell / 2 + 3.5) + ',' + (y0 - 5) + ') rotate(-90)'}, name, svg);
    });
    var cg = S('g', {'class': 'cells'}, svg), rects = {};
    function step(v) {
      if (isNaN(v)) { return 'qn'; }
      var s = 1 - v / hm.med;
      return s < 0.05 ? 'q0' : 'q' + Math.min(12, Math.floor(s / 0.05 + 1e-9));
    }
    for (i = 0; i < n; i++) {
      for (j = 0; j < n; j++) {
        if (i === j) { continue; }
        rects[i * n + j] = S('rect', {x: x0 + j * cell, y: y0 + i * cell, width: cell - gap, height: cell - gap, 'class': step(bw[i * n + j])}, cg);
      }
    }
    var hl = S('rect', {width: cell + 2, height: cell + 2, 'class': 'hlc', visibility: 'hidden'}, svg);
    var cur = [0, 1];
    function tipRows(a, b) {
      var v = bw[a * n + b], rows = [];
      if (isNaN(v)) { return [{v: 'no data'}, {l: hm.nodes[a] + ' ↔ ' + hm.nodes[b]}]; }
      rows.push({v: fmt(v, 'MB/s')});
      rows.push({l: hm.nodes[a] + ' ↔ ' + hm.nodes[b]});
      rows.push({l: Math.round(100 * v / hm.med) + '% of the ' + hm.group + ' median ' + fmt(hm.med, 'MB/s')});
      if (!isNaN(lt[a * n + b])) { rows.push({l: 'latency ' + fmt(lt[a * n + b], 'us')}); }
      return rows;
    }
    function mark(a, b) {
      hl.setAttribute('x', x0 + b * cell - 1); hl.setAttribute('y', y0 + a * cell - 1); hl.setAttribute('visibility', 'visible');
    }
    function clear() { hl.setAttribute('visibility', 'hidden'); hideTip(); }
    var first = rects[cur[0] * n + cur[1]];
    if (first) { first.setAttribute('tabindex', '0'); first.setAttribute('aria-label', hm.nodes[0] + ' and ' + hm.nodes[1]); }
    svg.addEventListener('pointermove', function (e) {
      var pt = svgPoint(svg, e), b = Math.floor((pt.x - x0) / cell), a = Math.floor((pt.y - y0) / cell);
      if (a < 0 || b < 0 || a >= n || b >= n || a === b) { clear(); return; }
      mark(a, b); showTip(tipRows(a, b), e.clientX, e.clientY);
    });
    svg.addEventListener('pointerleave', clear);
    function focusCell(a, b) {
      var old = rects[cur[0] * n + cur[1]], r = rects[a * n + b];
      if (!r) { return; }
      if (old && old !== r) { old.setAttribute('tabindex', '-1'); }
      r.setAttribute('tabindex', '0');
      r.setAttribute('aria-label', hm.nodes[a] + ' and ' + hm.nodes[b]);
      cur = [a, b];
      r.focus();
    }
    cg.addEventListener('focusin', function (e) {
      var r = e.target, a = cur[0], b = cur[1];
      mark(a, b); tipAt(r, tipRows(a, b));
    });
    cg.addEventListener('focusout', clear);
    cg.addEventListener('keydown', function (e) {
      var a = cur[0], b = cur[1];
      if (e.key === 'ArrowRight') { b++; } else if (e.key === 'ArrowLeft') { b--; }
      else if (e.key === 'ArrowDown') { a++; } else if (e.key === 'ArrowUp') { a--; }
      else if (e.key === 'Escape') { clear(); return; } else { return; }
      e.preventDefault();
      if (a === b) { if (e.key === 'ArrowRight') { b++; } else if (e.key === 'ArrowLeft') { b--; } else if (e.key === 'ArrowDown') { a++; } else { a--; } }
      if (a >= 0 && b >= 0 && a < n && b < n) { focusCell(a, b); }
    });
  }

  // ---- OSU curves: one line per MPI, log-log, legend + end labels + crosshair readout
  function renderLines(host, ch) {
    var W = Math.max(260, host.clientWidth), H = 236;
    var m = {l: 42, r: 66, t: 10, b: 34};
    var svg = S('svg', {width: W, height: H, viewBox: '0 0 ' + W + ' ' + H, tabindex: '0', role: 'img',
      'aria-label': ch.title + '; arrow keys move the readout'}, host);
    var sizes = [], vals = [];
    ch.series.forEach(function (s) {
      s.pts = s.pts.filter(function (p) { return p[0] > 0 && p[1] > 0 && isFinite(p[0]) && isFinite(p[1]); });
      s.pts.forEach(function (p) { if (sizes.indexOf(p[0]) < 0) { sizes.push(p[0]); } vals.push(p[1]); });
    });
    sizes.sort(function (a, b) { return a - b; });
    if (!sizes.length || !vals.length) { T('text', {x: m.l, y: 30, 'class': 'unit'}, 'no data', svg); return; }
    var x = sizes.length > 1 ? logsc(sizes[0], sizes[sizes.length - 1], m.l, W - m.r) : function () { return (m.l + W - m.r) / 2; };
    var ylo = Math.min.apply(null, vals) / 1.35, yhi = Math.max.apply(null, vals) * 1.35;
    var y = logsc(ylo, yhi, H - m.b, m.t);
    var gg = S('g', {'class': 'grid'}, svg), tg = S('g', {'class': 'tick'}, svg);
    logTicks(ylo, yhi, 6).forEach(function (t) {
      S('line', {x1: m.l, x2: W - m.r, y1: y(t), y2: y(t)}, gg);
      T('text', {x: m.l - 6, y: y(t) + 3.5, 'text-anchor': 'end'}, compact(t), tg);
    });
    var span = Math.log(sizes[sizes.length - 1] / sizes[0]) / Math.LN2, every = 1;
    var maxT = Math.max(3, Math.floor((W - m.l - m.r) / 46));
    while (isFinite(span) && span / every > maxT && every < 1e6) { every *= 2; }
    sizes.forEach(function (sz) {
      var k = Math.round(Math.log(sz / sizes[0]) / Math.LN2);
      if (k % every !== 0) { return; }
      S('line', {x1: x(sz), x2: x(sz), y1: m.t, y2: H - m.b}, gg);
      T('text', {x: x(sz), y: H - m.b + 15, 'text-anchor': 'middle'}, bytes(sz), tg);
    });
    S('line', {x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, 'class': 'axis'}, svg);
    T('text', {x: W - m.r, y: H - 3, 'text-anchor': 'end', 'class': 'unit'}, 'message size (bytes)', svg);
    T('text', {x: m.l - 6, y: m.t - 1, 'text-anchor': 'end', 'class': 'unit'}, ch.unit, svg);
    var ends = [];
    ch.series.forEach(function (s) {
      var pts = s.pts.filter(function (p) { return p[1] > 0; });
      if (!pts.length) { return; }
      var d = pts.map(function (p, i) { return (i ? 'L' : 'M') + r1(x(p[0])) + ' ' + r1(y(p[1])); }).join('');
      S('path', {d: d, 'class': 'ln c' + s.slot}, svg);
      var last = pts[pts.length - 1];
      ends.push({s: s, x: x(last[0]), y: y(last[1]), ly: y(last[1])});
    });
    ends.sort(function (a, b) { return a.y - b.y; });
    for (var k = 1; k < ends.length; k++) { if (ends[k].ly - ends[k - 1].ly < 13) { ends[k].ly = ends[k - 1].ly + 13; } }
    var over = ends.length ? ends[ends.length - 1].ly - (H - m.b - 2) : 0;
    if (over > 0) { ends.forEach(function (e) { e.ly -= over; }); }
    ends.forEach(function (e) {
      var lx = W - m.r + 8;
      if (Math.abs(e.ly - e.y) > 2 || lx - e.x > 6) { S('line', {x1: e.x + 4, y1: e.y, x2: lx - 3, y2: e.ly, 'class': 'lead'}, svg); }
      T('text', {x: lx, y: e.ly + 3.5, 'class': 'el'}, e.s.mpi, svg);
    });
    var xh = S('line', {y1: m.t, y2: H - m.b, 'class': 'xh', visibility: 'hidden'}, svg);
    var dots = ch.series.map(function (s) { return S('circle', {r: 4.5, 'class': 'ep f' + s.slot, visibility: 'hidden'}, svg); });
    var idx = sizes.length - 1;
    function readout(i) {
      var sz = sizes[i];
      xh.setAttribute('x1', x(sz)); xh.setAttribute('x2', x(sz)); xh.setAttribute('visibility', 'visible');
      var rows = [{v: bytes(sz) + ' bytes'}];
      ch.series.forEach(function (s, si) {
        var hit = null;
        s.pts.forEach(function (p) { if (p[0] === sz) { hit = p; } });
        if (hit && hit[1] > 0) {
          dots[si].setAttribute('cx', x(sz)); dots[si].setAttribute('cy', y(hit[1])); dots[si].setAttribute('visibility', 'visible');
          rows.push({key: s.slot, v: fmt(hit[1], ch.unit), l: s.mpi});
        } else {
          dots[si].setAttribute('visibility', 'hidden');
          rows.push({key: s.slot, v: 'no data', l: s.mpi});
        }
      });
      return rows;
    }
    function clear() { xh.setAttribute('visibility', 'hidden'); dots.forEach(function (d) { d.setAttribute('visibility', 'hidden'); }); hideTip(); }
    function nearest(px) {
      var best = 0, bd = Infinity;
      sizes.forEach(function (sz, i) { var d = Math.abs(x(sz) - px); if (d < bd) { bd = d; best = i; } });
      return best;
    }
    svg.addEventListener('pointermove', function (e) {
      var pt = svgPoint(svg, e);
      if (pt.x < m.l - 10 || pt.x > W - m.r + 10) { clear(); return; }
      idx = nearest(pt.x);
      showTip(readout(idx), e.clientX, e.clientY);
    });
    svg.addEventListener('pointerleave', clear);
    svg.addEventListener('focus', function () { var rows = readout(idx), b = svg.getBoundingClientRect(); showTip(rows, b.left + x(sizes[idx]), b.top + m.t); });
    svg.addEventListener('blur', clear);
    svg.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        e.preventDefault();
        idx = Math.max(0, Math.min(sizes.length - 1, idx + (e.key === 'ArrowRight' ? 1 : -1)));
        var rows = readout(idx), b = svg.getBoundingClientRect();
        showTip(rows, b.left + x(sizes[idx]), b.top + m.t);
      } else if (e.key === 'Escape') { clear(); }
    });
  }

  var RENDER = {strip: renderStrip, base: renderBase, heat: renderHeat, lines: renderLines};
  function render(fig) {
    var host = fig.querySelector('.plot');
    if (!host || host.closest('[hidden]')) { return; }
    var kind = fig.getAttribute('data-kind'), item = D[kind][+fig.getAttribute('data-idx')];
    host.textContent = '';
    fig.setAttribute('data-w', host.parentNode.clientWidth);
    RENDER[kind](host, item);
  }
  var figs = Array.prototype.slice.call(document.querySelectorAll('figure[data-kind]'));
  figs.forEach(render);
  var timer = null;
  window.addEventListener('resize', function () {
    clearTimeout(timer);
    timer = setTimeout(function () {
      figs.forEach(function (f) {
        var host = f.querySelector('.plot');
        if (host && +f.getAttribute('data-w') !== host.parentNode.clientWidth) { render(f); }
      });
    }, 150);
  });
  document.querySelectorAll('button.tbl-toggle').forEach(function (b) {
    b.addEventListener('click', function () {
      var fig = b.closest('figure'), tbl = fig.querySelector('.tablewrap'), plot = fig.querySelector('.plotwrap');
      var showTable = tbl.hidden;
      tbl.hidden = !showTable;
      plot.hidden = showTable;
      b.setAttribute('aria-expanded', showTable ? 'true' : 'false');
      b.textContent = showTable ? 'Chart' : 'Table';
      if (!showTable) { render(fig); }
    });
  });
})();
'''


def esc(s):
    return html.escape('' if s is None else str(s), quote=True)


def icon(status, label=None):
    title = ' aria-hidden="true"' if label is None else ' role="img" aria-label="%s"' % esc(label)
    return '<svg class="ico"%s><use href="#i-%s"/></svg>' % (title, status)


def badge(category):
    st = STATUS[SEVERITY[category]]
    return '<span class="badge">%s%s</span>' % (icon(st), esc(category))


ICON_DEFS = '''<svg width="0" height="0" style="position:absolute" aria-hidden="true" focusable="false"><defs>
<symbol id="i-critical" viewBox="0 0 16 16"><path d="M5.1 1h5.8L15 5.1v5.8L10.9 15H5.1L1 10.9V5.1z" style="fill:var(--critical)"/><path d="M5.6 5.6l4.8 4.8M10.4 5.6l-4.8 4.8" style="stroke:#fff" stroke-width="1.8" stroke-linecap="round"/></symbol>
<symbol id="i-serious" viewBox="0 0 16 16"><path d="M8 1.3l7 12.9H1z" style="fill:var(--serious)"/><path d="M8 6v3.6" style="stroke:#0b0b0b" stroke-width="1.8" stroke-linecap="round"/><circle cx="8" cy="12" r="1" style="fill:#0b0b0b"/></symbol>
<symbol id="i-warning" viewBox="0 0 16 16"><path d="M8 .9L15.1 8 8 15.1.9 8z" style="fill:var(--warning)"/><path d="M8 4.6v4" style="stroke:#0b0b0b" stroke-width="1.8" stroke-linecap="round"/><circle cx="8" cy="11.1" r="1" style="fill:#0b0b0b"/></symbol>
<symbol id="i-info" viewBox="0 0 16 16"><circle cx="8" cy="8" r="6.6" style="fill:none;stroke:var(--muted)" stroke-width="1.6"/><path d="M8 7.2v4.1" style="stroke:var(--ink-2)" stroke-width="1.8" stroke-linecap="round"/><circle cx="8" cy="4.9" r="1" style="fill:var(--ink-2)"/></symbol>
<symbol id="i-good" viewBox="0 0 16 16"><circle cx="8" cy="8" r="7" style="fill:var(--good)"/><path d="M4.7 8.2l2.2 2.2 4.4-4.6" style="fill:none;stroke:#fff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></symbol>
</defs></svg>'''

KEY_DOT = '<svg width="12" height="12" aria-hidden="true"><circle cx="6" cy="6" r="5" style="fill:var(--%s)"/></svg>'
KEY_TICK = '<svg width="6" height="14" aria-hidden="true"><line x1="3" x2="3" y1="1" y2="13" style="stroke:var(--ink)" stroke-width="2" stroke-linecap="round"/></svg>'
KEY_ZONE = '<svg width="18" height="12" aria-hidden="true"><rect width="18" height="12" style="fill:%s"/></svg>'


def base_key(neg, pos):
    '''Legend of a change-vs-baseline chart whose tolerance band runs from neg to pos (in % change).'''
    band = '&plusmn;%.0f%%' % pos if round(-neg, 6) == round(pos, 6) else '&minus;%.0f%% to +%.0f%%' % (-neg, pos)
    return ('<div class="key"><span>%s slower than &minus;%.0f%%</span><span>%s within %s</span>'
            '<span>%s faster than +%.0f%%</span><span>%s tolerance band</span></div>'
            % (KEY_DOT % 'div-neg', -neg, KEY_DOT % 'muted', band, KEY_DOT % 'div-pos', pos,
               KEY_ZONE % 'var(--div-mid)'))


def table_html(head, rows, num_cols=()):
    out = ['<table><thead><tr>']
    for i, h in enumerate(head):
        out.append('<th%s>%s</th>' % (' class="num"' if i in num_cols else '', esc(h)))
    out.append('</tr></thead><tbody>')
    for r in rows:
        out.append('<tr>' + ''.join('<td%s>%s</td>' % (' class="num"' if i in num_cols else '', esc(c))
                                    for i, c in enumerate(r)) + '</tr>')
    out.append('</tbody></table>')
    return ''.join(out)


def figure(kind, idx, fid, title, sub, table, key='', wide=False):
    plot = '<div class="plot"></div>'
    if wide:
        plot = '<div class="heatscroll">%s</div>' % plot
    return ('<figure class="card chart" id="%s" data-kind="%s" data-idx="%d"><div class="fig-head"><div><h3>%s</h3>'
            '<p class="sub">%s</p></div><button type="button" class="tbl-toggle" aria-expanded="false" '
            'aria-controls="%s-t">Table</button></div><div class="plotwrap">%s%s</div>'
            '<div class="tablewrap" id="%s-t" hidden>%s</div></figure>'
            % (esc(fid), kind, idx, esc(title), sub, esc(fid), key, plot, esc(fid), table))


def strip_data(A, P, metric, what):
    groups, stats = A['strips'].get(metric, ({}, {}))
    if not groups:
        return None
    of = A['of']
    flagged = {}
    for p in P:
        if p['category'] == 'OUTLIER' and p['metric'] == metric:
            flagged[(p['node'], p.get('device', ''))] = 'OUTLIER'
    for p in A['flag_rows']:
        if p['metric'] == metric:
            flagged[(p['node'], p.get('device', ''))] = 'OUTLIER'
    for p in P:
        base = p.get('base_metric') or p['metric']
        if p['category'] == 'TOO_SLOW' and base == metric and p['expected'] is not None:
            for v, node, dev, extra in sum(groups.values(), []):
                if node == p['node'] and v < p['expected']:
                    flagged[(node, dev)] = 'TOO_SLOW'
    higher = METRICS[metric][2]
    rows, table = [], []
    for g in sorted(groups, key=natkey):
        st = stats.get(g, {})
        med = st.get('median')
        src = st.get('src') or ''          # "run <id>": too few values here, the parent run's median is used
        pts = []
        for v, node, dev, extra in sorted(groups[g], key=lambda t: t[0]):
            label = dev_label(dev, extra) if dev else ''
            flag = flagged.get((node, dev), '')
            pct = round(100.0 * v / med, 2) if med else None
            pts.append([round(v, 4), node, label, flag, pct])
            table.append([g + (' (median from %s)' % src if src else ''), node, label, fmt(v),
                          '%.0f%%' % pct if pct is not None else '', flag])
        rows.append({'g': g, 'n': len(pts), 'med': med, 'checked': bool(st.get('checked')), 'pts': pts,
                     'src': src, 'srcn': st.get('src_n', 0)})
    # the shared axis is "% of the group median", the scale the OUTLIER rule uses
    return {'rows': rows, 'unit': METRICS[metric][1], 'lower': not higher, 'what': what,
            'cut': round(100.0 * outlier_limit(metric, 1.0, of), 2), 'title': METRICS[metric][3]}, table


def write_html(path, run, A, P, issues, counts, rerun, baseline_id, base_chart, of, hd, mpi_text=''):
    data = {'strip': [], 'base': [], 'heat': [], 'lines': []}
    out = []
    gen = datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')
    reservation = run.inv.get('reservation') or sol.env('RESERVATION', '') or '-'
    sess = []
    if run.node_info:
        sess.append('node stage %s' % run.node_info.get('uuid', '?'))
    if run.mpi_info:
        sess.append('MPI stage %s' % run.mpi_info.get('uuid', '?'))
    pct_of = '%.0f%%' % (100 * of)
    out.append('<header><h1>Sol pre-release check</h1>')
    meta = 'Run <b>%s</b> &middot; reservation <b>%s</b> &middot; generated %s' % (esc(run.run_id), esc(reservation),
                                                                                    esc(gen))
    if baseline_id:
        meta += ' &middot; baseline <b>%s</b>' % esc(baseline_id)
    if A.get('parent'):
        meta += ' &middot; parent <b>%s</b>' % esc(A['parent']['id'])
    out.append('<p class="meta">%s</p>' % meta)
    out.append('<p class="meta">ReFrame sessions: %s</p>' % esc('; '.join(sess) or 'none found'))
    toc = [('problems', 'Problems'), ('perf', 'Performance by group'), ('fabric', 'Fabric'), ('mpi', 'MPI candidates')]
    if baseline_id:
        toc.append(('baseline', 'Change vs baseline'))
    out.append('<nav class="toc">%s</nav></header>' % ''.join('<a href="#%s">%s</a>' % t for t in toc))
    for w in run.warnings:
        out.append('<div class="verdict">%s<div>%s</div></div>' % (icon('warning', 'warning'), esc(w)))
    bad_nodes = sorted(set(p['node'] for p in P if p['node'] and '+' not in p['node']
                           and p['severity'] <= EXIT_SEVERITY), key=natkey)
    if not run.have_node_stage:
        out.append('<div class="verdict">%s<div><b>No node-stage results</b> (stage_node.json missing or unreadable): '
                   'nothing was verified; report.py exits with 1.</div></div>' % icon('critical', 'not verified'))
    elif bad_nodes:
        unav = [h for h in bad_nodes if not (run.nodes.get(h) or {}).get('testable')]
        split = ' (%d unavailable, %d tested)' % (len(unav), len(bad_nodes) - len(unav)) if unav else ''
        out.append('<div class="verdict">%s<div><b>%d node%s with problems of severity 1-5</b>%s: UNAVAILABLE to '
                   'TOO_SLOW. Fix or exclude them before release; report.py exits with 1.%s</div></div>'
                   % (icon('critical', 'not ready'), len(bad_nodes), '' if len(bad_nodes) == 1 else 's', split,
                      ' The MPI stage failed too: %s.' % esc(mpi_text) if mpi_text else ''))
    elif mpi_text:
        out.append('<div class="verdict">%s<div><b>No node problems of severity 1-5, but the MPI stage failed</b>: '
                   '%s (see MPI candidates). report.py exits with 1.</div></div>'
                   % (icon('critical', 'not ready'), esc(mpi_text)))
    else:
        out.append('<div class="verdict">%s<div><b>No node problems of severity 1-5.</b> Findings below are '
                   'informational.</div></div>' % icon('good', 'ok'))
    inv_note = 'unknown: no nodes.json'
    if counts['not_tested'] is None:
        not_tested = inv_note
    elif counts['untestable']:
        not_tested = '%d unavailable in Slurm, %d cannot run the test' % (counts['unavailable'], counts['untestable'])
    else:
        not_tested = 'unavailable in Slurm'
    tiles = [('Testable' if counts['not_tested'] is not None else 'Tested', counts['testable'], 'nodes in this run'),
             ('Passed', counts['passed'], 'no findings' if run.have_node_stage else 'no node-stage results'),
             ('With problems', counts['problems'], 'see the table below'),
             ('Not tested', counts['not_tested'], not_tested),
             ('Excluded', counts['excluded'], 'EXCLUDE_NODES' if counts['excluded'] is not None else inv_note)]
    if counts['not_selected']:
        tiles.append(('Not selected', counts['not_selected'], 'outside --only'))
    out.append('<section class="tiles" aria-label="node counts">%s</section>' % ''.join(
        '<div class="tile"><div class="l">%s</div><div class="v">%s</div><div class="d">%s</div></div>'
        % (esc(l), 'n/a' if v is None else '{:,}'.format(v), esc(d)) for l, v, d in tiles))

    # problems
    rows = [p for p in P if p['node'] and not p.get('systemic')]
    out.append('<section id="problems"><h2>Problem nodes</h2><p class="lead">Worst node first. Severity order: '
               'UNAVAILABLE, JOB_FAILED, WRONG_RESULT, MISMATCH, TOO_SLOW, OUTLIER, SLOW_LINK, SLOWER_THAN_BASELINE, '
               'NOT_MEASURED (information only). Findings shared by many nodes are grouped under Systemic issues.</p>')
    if rows:
        body = ''.join('<tr><td>%s</td><td class="mono">%s</td><td>%s</td><td class="reason">%s</td></tr>'
                       % (badge(p['category']), esc(p['node']), esc(p['check']), esc(p['reason'])) for p in rows)
        out.append('<div class="card tscroll"><table class="ptable"><thead><tr><th>Category</th><th>Node</th>'
                   '<th>Check</th><th>Reason</th></tr></thead><tbody>%s</tbody></table></div>' % body)
    else:
        out.append('<div class="card">%s No problem nodes.</div>' % icon('good'))
    out.append('<h3 style="margin:20px 0 8px">Systemic issues</h3>')
    if issues:
        items = []
        for s in issues:
            unit = 'links among %d nodes' % len(s['hosts']) if s['category'] == 'SLOW_LINK' else 'nodes'
            det = ''.join('<tr><td class="mono">%s</td><td class="reason">%s</td></tr>'
                          % (esc(r['node']), esc(r['reason'])) for r in s['items'])
            items.append('<div class="sys">%s<div style="min-width:0"><div class="t">[%s] %s on %d %s</div>'
                         '<div class="hosts">%s</div><div>%s</div><details><summary>%d rows</summary>'
                         '<table class="ptable"><tbody>%s</tbody></table></details></div></div>'
                         % (icon(STATUS[s['severity']]), esc(s['check']), esc(s['category']), s['count'], unit,
                            esc(hostlist(s['hosts'])), esc(s['reason']), len(s['items']), det))
        out.append('<div class="card">%s</div>' % ''.join(items))
    else:
        out.append('<div class="card small">None.</div>')
    out.append('</section>')

    # performance by group
    from_parent = sorted(set(st['src'] for _, stats in A['strips'].values() for st in stats.values()
                             if st.get('src')))
    out.append('<section id="perf"><h2>Performance by group</h2><p class="lead">Each row is a comparison group and '
               'each dot a node or a GPU, placed at its percentage of the group median; the median itself is in the '
               'row label. The tick marks the median (100%%) and the shaded area is the outlier zone: below %s, or '
               'above %.0f%% for MPI_Init, where lower is better and one cold start per node varies a lot. Flagged '
               'dots are red and labelled. Hover, or tab to a chart and use the arrow keys, for values; the Table '
               'button lists them all.%s</p>'
               % (pct_of, 100.0 * outlier_limit('mpi_init', 1.0, of),
                  ' A group with fewer than %d values in this run is placed against the median of %s instead, as '
                  'its label says.' % (MIN_GROUP, esc(', '.join(from_parent))) if from_parent else ''))
    key = ('<div class="key"><span>%s node or GPU</span><span>%s flagged</span><span>%s group median</span>'
           '<span>%s outlier zone</span></div>' % (KEY_DOT % 's1', KEY_DOT % 'critical', KEY_TICK,
                                                   KEY_ZONE % 'var(--zone)'))
    specs = [('triad', 'node', 'STREAM triad by CPU group', 'One dot per node; CPU group = CPU model, cores, memory.'),
             ('mpi_init', 'node', 'MPI_Init time by CPU group', 'One rank per core; time on rank 0. Lower is '
                                                                   'better.'),
             ('sgemm', 'GPU', 'GPU SGEMM (FP32, no TF32) by GPU type', 'One dot per GPU or MIG device, pooled by '
                                                                       'name and memory.'),
             ('dgemm', 'GPU', 'GPU DGEMM (FP64) by GPU type', 'One dot per GPU or MIG device.'),
             ('hbm', 'GPU', 'GPU memory bandwidth by GPU type', 'Triad kernel on device memory.'),
             ('h2d', 'GPU', 'PCIe host to device by GPU type', 'Pinned 256 MiB copies; a GPU at half speed '
                                                               'usually sits in a degraded (x8) link.'),
             ('d2h', 'GPU', 'PCIe device to host by GPU type', 'Pinned 256 MiB copies.'),
             ('link_bw_median_node', 'node', 'Median link bandwidth per node by MPI group',
              'Each node\'s median bidirectional bandwidth (16 MiB messages) to every other node in the group.')]
    for metric, what, title, sub in specs:
        res = strip_data(A, P, metric, what)
        if not res:
            continue
        d, trows = res
        d['title'] = title
        idx = len(data['strip'])
        data['strip'].append(d)
        unit = METRICS[metric][1]
        out.append(figure('strip', idx, 'perf-' + metric, title, esc(sub), table_html(
            ['Group', 'Node', 'Device', 'Value (%s)' % unit, '% of median', 'Flag'], trows, (3, 4)), key))
    out.append('</section>')

    # fabric
    out.append('<section id="fabric"><h2>Fabric</h2><p class="lead">All-pairs bidirectional bandwidth per MPI group, '
               'each pair measured alone, so a slow pair is not contention. Each cell is one node pair, coloured by '
               'how far it falls below the group median in 5% steps (the scale under each chart); cells at or above '
               'the median, or within 5% of it, get the first colour of the scale. A row and column that stand out '
               'from the rest are a slow node; a single cell pair that stands out is a slow link (cable or switch '
               'port).</p>')
    ramp = ('<div class="ramp" aria-label="colour scale"><div class="bar">%s%s</div><span class="nd"><i></i>no data'
            '</span></div>' % (''.join('<i class="q%d" style="background:var(--q%d)"></i>' % (k, k) for k in range(13)),
                               ''.join('<b style="grid-column:%d">%s</b>' % (k + 1, lab) for k, lab in
                                       ((0, 'within 5%'), (4, '&minus;20%'), (8, '&minus;40%'),
                                        (11, '&le;&minus;60%')))))
    have_heat = False
    for g in [x['name'] for x in sol.mpi_groups()] + sorted(set(A['pairs']) - set(x['name'] for x in sol.mpi_groups())):
        d = A['pairs'].get(g)
        if not d or not d.get('median'):
            continue
        have_heat = True
        names = sorted(d['node_median'], key=natkey)
        index = dict((h, i) for i, h in enumerate(names))
        flag = [index[p['node']] for p in P if p['category'] in ('OUTLIER', 'TOO_SLOW') and p['group'] == g
                and p['metric'] == 'link_bw_median_node' and p['node'] in index]
        flag += [index[p['node']] for p in A['flag_rows'] if p['group'] == g and p['metric'] == 'link_bw_median_node'
                 and p['node'] in index]
        flag += [index[h] for p in P if p['category'] == 'SLOW_LINK' and p['group'] == g for h in p.get('hosts') or []
                 if h in index]
        pairs = [[index[a], index[b], round(bw, 1), round(lat, 2) if lat is not None else None]
                 for a, b, bw, lat in d['pairs'] if bw is not None and a in index and b in index]
        idx = len(data['heat'])
        data['heat'].append({'group': g, 'nodes': names, 'med': d['median'], 'flag': sorted(set(flag)),
                             'pairs': pairs})
        slow = [p for p in P if p['group'] == g and p['category'] == 'SLOW_LINK']
        sub = ('%d nodes, %s pairs, median %s. %s' % (len(names), '{:,}'.format(len(pairs)), fmt(d['median'], 'MB/s'),
                                                       '%d slow link%s.' % (len(slow), '' if len(slow) == 1 else 's')
                                                       if slow else 'No slow links.'))
        trows = [[a, b, fmt(bw), '%.0f%%' % (100 * bw / d['median']), fmt(lat)] for a, b, bw, lat in
                 sorted(d['pairs'], key=lambda t: (t[2] if t[2] is not None else 0))]
        out.append(figure('heat', idx, 'fabric-' + g, '%s pairwise bandwidth' % g, esc(sub),
                          table_html(['Node A', 'Node B', 'bibw (MB/s)', '% of median', 'latency (us)'], trows,
                                     (2, 3, 4)), ramp, wide=True))
    if not have_heat:
        out.append('<div class="card small">No pairwise data in this run.</div>')
    out.append('</section>')

    # MPI candidates
    out.append('<section id="mpi"><h2>MPI candidates</h2><p class="lead">OSU micro-benchmarks 7.5.2 per MPI group, '
               'one line per MPI (log-log). Point-to-point runs between the first two usable nodes; collectives use '
               'up to the group\'s collective node count with one rank per core.</p>')
    mrows = A['mpi_rows']
    builds = [r for r in mrows if r['test'] == 'build']
    if builds:
        out.append('<div class="card">%s</div>' % ''.join(
            '<div class="verdict" style="margin:0">%s<div><b>OSU build failed for %s</b>: %s</div></div>'
            % (icon('critical', 'failed'), esc(r['mpi']), esc(r['reason'])) for r in builds))
    if not run.have_mpi_stage:
        out.append('<div class="card small">MPI stage not run (no stage_mpi.json).</div>')
    groups_seen = [x['name'] for x in sol.mpi_groups()]
    groups_seen += sorted(set(r['group'] for r in mrows if r['group'] not in groups_seen and r['group'] != '-'))
    titles = {'pt2pt/bw': ('Point-to-point bandwidth', 'MB/s'), 'pt2pt/latency': ('Point-to-point latency', 'us'),
              'collective/allreduce': ('Allreduce latency', 'us'), 'collective/alltoall': ('Alltoall latency', 'us'),
              'gpudirect/bw': ('GPU-direct bandwidth', 'MB/s'), 'gpudirect/latency': ('GPU-direct latency', 'us')}
    for g in groups_seen:
        grows = [r for r in mrows if r['group'] == g]
        if not grows:
            continue
        out.append('<div class="card"><h3>%s</h3>' % esc(g))
        mpis = [k for k in MPI_ORDER if any(r['mpi'] == k for r in grows)]
        mpis += sorted(set(r['mpi'] for r in grows if r['mpi'] not in mpis and r['test'] not in ('pairwise', 'all')))
        tests = sorted(set(r['test'] for r in grows), key=lambda t: TEST_ORDER.index(t) if t in TEST_ORDER else 99)
        head = '<tr><th>Test</th>%s</tr>' % ''.join('<th>%s</th>' % esc(k) for k in (mpis or ['MPI']))
        body = []
        for t in tests:
            cells = []
            for k in (mpis if t not in ('pairwise', 'all') else [None]):
                r = next((x for x in grows if x['test'] == t and (k is None or x['mpi'] == k)), None)
                dl = ' data-label="%s"' % esc(k) if k else ''
                if r is None:
                    cells.append('<td class="small"%s>-</td>' % dl)
                    continue
                res = {'pass': ('good', 'PASS'), 'fail': ('critical', 'FAIL'), 'fail_deps': ('warning', 'SKIP'),
                       'skip': ('warning', 'SKIP'), 'not run': ('info', 'NOT RUN')}.get(r['result'], ('warning', str(r['result']).upper()))
                span = ' colspan="%d"' % max(1, len(mpis)) if k is None else ''
                lines = []
                if k is None and r['mpi'] not in ('-', ''):
                    mod = sol.env('MOD_NODE_MPI') if r['mpi'] == 'node-mpi' else ''
                    lines.append('MPI: %s%s' % (r['mpi'], ' (%s)' % mod if mod else ''))
                lines += mpi_numbers(r).split('  ') if r['result'] == 'pass' else [mpi_numbers(r)]
                cells.append('<td%s%s><span class="res">%s%s</span><div class="small">%s</div></td>'
                             % (span, dl, icon(res[0]), res[1], '<br>'.join(esc(x) for x in lines if x)))
            body.append('<tr><td>%s</td>%s</tr>' % (esc(t), ''.join(cells)))
        out.append('<div class="tscroll"><table class="mpi-table"><thead>%s</thead><tbody>%s</tbody></table></div>'
                   % (head, ''.join(body)))
        osu = A['osu'].get(g, {})
        figs = []
        for t in TEST_ORDER:
            if t not in titles:
                continue
            series = []
            for k in mpis:
                d = osu.get((t, k))
                r = next((x for x in grows if x['test'] == t and x['mpi'] == k), None)
                if d is None and r is None:
                    continue
                series.append({'mpi': k, 'slot': SERIES_SLOT.get(k, 0),
                               'pts': [[sz, round(v, 3)] for sz, v in (d['rows'] if d else []) if sz > 0 and v > 0],
                               'ok': bool(r and r['result'] == 'pass')})
            if not series or not any(s['pts'] for s in series):
                continue
            title, unit = titles[t]
            idx = len(data['lines'])
            data['lines'].append({'title': '%s, %s' % (title, g), 'unit': unit, 'series': series})
            legend = ''.join('<span><i class="lkey k%d"></i>%s%s</span>' % (
                s['slot'], esc(s['mpi']), '' if s['ok'] else (' (failed, partial data)' if s['pts'] else
                                                              ' (no data)')) for s in series)
            sizes = sorted(set(sz for s in series for sz, _ in s['pts']))
            trows = []
            for sz in sizes:
                row = [str(sz)]
                for s in series:
                    v = next((v for z, v in s['pts'] if z == sz), None)
                    row.append(fmt(v) if v is not None else '')
                trows.append(row)
            figs.append(figure('lines', idx, 'osu-%s-%s' % (g, t.replace('/', '-')), title,
                               esc('%s, %s' % (unit, 'log-log')), table_html(['Size (bytes)'] + [s['mpi'] for s in
                                                                                                  series], trows,
                                                                            tuple(range(1, len(series) + 1))),
                               '<div class="key">%s</div>' % legend))
        if figs:
            out.append('<div class="osu-grid">%s</div>' % ''.join(figs))
        out.append('</div>')
    out.append('</section>')

    # baseline
    if baseline_id:
        out.append('<section id="baseline"><h2>Change vs baseline</h2><p class="lead">Each dot is the same node or '
                   'GPU in this run compared with run %s. Negative is slower (for MPI_Init, a longer time counts as '
                   'slower). The grey band is the tolerated change of &plusmn;%.0f%%, for MPI_Init up to %gx the '
                   'baseline time (&minus;%.0f%%); red dots beyond it are flagged as SLOWER_THAN_BASELINE and '
                   'labelled, blue dots got faster.</p>' % (esc(baseline_id), 100 * hd, SLOW_FACTOR['mpi_init'],
                                                            100 * baseline_drop('mpi_init', hd)))
        for metric, what, title, sub in specs:
            pts = base_chart.get(metric)
            if not pts:
                continue
            groups = {}
            for pct, node, dev, now, before, grp in pts:
                groups.setdefault(grp, []).append([round(pct, 2), node, dev_label(dev, A['gpu_pci'].get(dev, ''))
                                                   if dev else '', now, before])
            rows = [{'g': g, 'pts': groups[g]} for g in sorted(groups, key=natkey)]
            idx = len(data['base'])
            # the band, in % change: HISTORY_DROP both ways, or SLOW_FACTOR slower or faster for the timings
            neg = -100 * baseline_drop(metric, hd)
            pos = 100 * (SLOW_FACTOR[metric] - 1) if metric in SLOW_FACTOR else 100 * hd
            data['base'].append({'title': title, 'unit': METRICS[metric][1], 'neg': neg, 'pos': pos, 'what': what,
                                 'rows': rows})
            trows = [[r['g'], p[1], p[2], fmt(p[4]), fmt(p[3]), '%+.1f%%' % p[0]] for r in rows
                     for p in sorted(r['pts'], key=lambda q: q[0])]
            out.append(figure('base', idx, 'base-' + metric, title + ': change vs baseline',
                              esc('%s matched with the baseline.' % '{:,}'.format(len(pts))),
                              table_html(['Group', 'Node', 'Device', 'Baseline', 'Now', 'Change'], trows, (3, 4, 5)),
                              base_key(neg, pos)))
        out.append('</section>')

    # footer
    out.append('<footer id="next"><h2>Next steps</h2>')
    if rerun:
        out.append('<p>Rerun the node check on the problem nodes (UNAVAILABLE to OUTLIER) after fixing them:</p>'
                   '<pre class="cmd">./run_prerelease.sh --only \'%s\'</pre>' % esc(hostlist(rerun)))
    else:
        out.append('<p>No node needs a rerun.</p>')
    out.append('<p class="small" style="margin-top:12px">Files next to this page: problems.csv (every finding), '
               'metrics.csv (every measured number, the long-term record). OUTLIER_FRACTION=%s, HISTORY_DROP=%s '
               '(MPI_Init: %gx the group median; MPI_Init and barrier: %gx the baseline), MAX_PENDING=%s, '
               'MPI_MAX_PENDING=%s.</p></footer>'
               % (esc('%g' % of), esc('%g' % hd), SLOW_FACTOR['mpi_init'], SLOW_FACTOR['barrier'],
                  esc(sol.env('MAX_PENDING', '30m')), esc(sol.env('MPI_MAX_PENDING', '4h'))))

    blob = json.dumps(data, separators=(',', ':'), allow_nan=False)
    blob = blob.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            '<title>Sol pre-release %s</title>\n'
            '<script>(function(){var t=new URLSearchParams(location.search).get("theme");'
            'if(t==="dark"||t==="light"){document.documentElement.setAttribute("data-theme",t);}})();</script>\n'
            '<style>%s</style>\n</head>\n<body class="viz-root">\n%s\n<main class="wrap">\n%s\n</main>\n'
            '<div id="tip" role="tooltip" aria-live="polite" hidden></div>\n'
            '<script type="application/json" id="report-data">%s</script>\n<script>%s</script>\n</body>\n</html>\n'
            % (esc(run.run_id), CSS, ICON_DEFS, '\n'.join(out), blob, JS))
    tmp = path + '.tmp'
    with open(tmp, 'w') as fp:
        fp.write(page)
    os.replace(tmp, path)


if __name__ == '__main__':
    sys.exit(main())
