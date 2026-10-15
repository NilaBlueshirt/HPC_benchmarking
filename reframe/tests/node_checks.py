# node_checks.py - SolNodeCheck: one whole-node Slurm job per testable node in nodes.json.
# The job runs src/node_check.sh (srun launch, device count, MPI hello, STREAM, GPU checks),
# which prints SOL_* lines. Sanity collects every problem as "[step] message" in one
# SanityError; performance reports the per-node numbers.
#
# Usage (run_prerelease.sh does this):
#   SOL_INVENTORY=results/<run>/nodes.json reframe -C config/sol_config.py -c tests/node_checks.py -r

import math
import os
import re
import shlex

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.backends import getlauncher
from reframe.core.builtins import parameter, run_after, run_before, sanity_function
from reframe.core.exceptions import SanityError
from reframe.core.logging import getlogger

import sol_common as sol

# perf variable, unit, SOL_* token, field, step it needs; GPU values are the minimum over devices
METRICS = [
    ('mpi_init', 's', 'SOL_MPI_HELLO', 'init_s', 'mpi'),
    ('barrier', 'us', 'SOL_MPI_HELLO', 'barrier_us', 'mpi'),
    ('triad', 'MB/s', 'SOL_STREAM', 'triad', 'stream'),
    ('copy', 'MB/s', 'SOL_STREAM', 'copy', 'stream'),
    ('sgemm_min', 'TFLOP/s', 'SOL_GPU_GEMM', 'sgemm_tflops', 'gpu'),
    ('dgemm_min', 'TFLOP/s', 'SOL_GPU_GEMM', 'dgemm_tflops', 'gpu'),
    ('hbm_min', 'GB/s', 'SOL_GPU_HBM', 'hbm_gbs', 'gpu'),
    ('h2d_min', 'GB/s', 'SOL_GPU_PCIE', 'h2d_gbs', 'gpu'),
    ('d2h_min', 'GB/s', 'SOL_GPU_PCIE', 'd2h_gbs', 'gpu'),
]

# lower is better: a lower-bound reference would flag fast nodes, so these never get one
LOWER_IS_BETTER = {'mpi_init', 'barrier'}

# Slurm end states that mean the job did not end normally
BAD_STATES = {'BOOT_FAIL', 'CANCELLED', 'DEADLINE', 'FAILED', 'NODE_FAIL', 'OUT_OF_MEMORY',
              'PREEMPTED', 'TIMEOUT'}


def _num(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _int(value):
    x = _num(value)
    return int(x) if x is not None and x == int(x) else None


def _clean(text):
    '''Free text from job output, safe inside "[step] msg; [step] msg" fail reasons.'''
    text = str(text).replace('[', '(').replace(']', ')').replace(';', ',')
    return ' '.join(text.split())


def _tag(value):
    return re.sub(r'[^a-z0-9_-]+', '_', str(value).lower()).strip('_') or 'unknown'


def _shell_value(value):
    '''env_vars are written unquoted as export K=V, so quote anything that is not a plain word.'''
    value = str(value)
    return value if re.match(r'^[A-Za-z0-9_./:+=,@%-]*$', value) else shlex.quote(value)


def _cores(node):
    # ReFrame treats num_tasks <= 0 as a flexible allocation, so never go below 1
    return max(1, _int(node.get('cores')) or 0)


_REFS = []


def _node_references():
    '''refs["SolNodeCheck"] from SOL_REFERENCES, read once; a broken file only costs the references.'''
    if not _REFS:
        try:
            refs = sol.load_references()
        except (OSError, ValueError) as err:
            getlogger().warning('SolNodeCheck: ignoring references file: %s' % err)
            refs = {}
        refs = refs.get('SolNodeCheck') if isinstance(refs, dict) else None
        _REFS.append(refs if isinstance(refs, dict) else {})
    return _REFS[0]


def _metric(path, token, field):
    '''Minimum of FIELD over the TOKEN lines of the job output.'''
    vals = [_num(f.get(field)) for t, f in sol.read_sol_lines(path) if t == token]
    vals = [v for v in vals if v is not None]
    if not vals:
        raise ValueError('no %s %s= value in %s' % (token, field, path))
    return min(vals)


@rfm.simple_test
class SolNodeCheck(rfm.RunOnlyRegressionTest):
    node = parameter(sol.testable_nodes(), fmt=lambda n: n['name'])
    descr = 'Sol pre-release check of one node (src/node_check.sh)'
    valid_systems = ['sol:compute']
    valid_prog_environs = ['builtin']
    sourcesdir = 'src'
    executable = 'bash'
    executable_opts = ['node_check.sh']
    exclusive_access = True
    time_limit = sol.env('NODE_TIME_LIMIT', '20m')
    max_pending_time = sol.env('MAX_PENDING', '30m')
    tags = {'node'}

    @run_after('init')
    def configure(self):
        n = self.node
        mods = sol.node_modules(n)
        self.num_tasks = _cores(n)
        self.num_tasks_per_node = _cores(n)
        self.tags = self.tags | {_tag(n['cpu_group']), _tag(n['accel']), _tag(n['arch'])}
        env = {
            'SOL_NODE': n['name'],
            'SOL_CORES': _cores(n),
            'SOL_ACCEL': n['accel'],
            'SOL_GPU_COUNT': n['gpu_count'],
            'SOL_MOD_GCC': mods['gcc'],
            'SOL_MOD_CUDA': mods['cuda'],
            'SOL_MOD_NODE_MPI': mods['node_mpi'],
            'SOL_SRUN_MPI': sol.env('SRUN_MPI_node_mpi', sol.env('SRUN_MPI', 'pmix')),
            'SOL_STREAM_ELEMS': sol.env('STREAM_ELEMS', '268435456'),
            'SOL_GPU_SECONDS': sol.env('GPU_SECONDS', '20'),
        }
        self.env_vars = {k: _shell_value(v) for k, v in env.items()}

        applies = {'mpi': bool(mods['node_mpi']), 'stream': True,
                   'gpu': n['accel'] == 'nvidia' and bool(mods['cuda'])}
        self.perf_variables = {
            name: sn.make_performance_function(_metric, unit, self.stdout, token, field)
            for name, unit, token, field, step in METRICS if applies[step]
        }

    @run_after('setup')
    def set_references(self):
        refs = _node_references()
        if not refs:
            return
        n = self.node
        low = -(1.0 - sol.env_float('OUTLIER_FRACTION', 0.90))
        cpu = refs.get(n['cpu_group']) or {}
        gpu = (refs.get('gpu:' + n['gpu_model']) if n.get('gpu_model') else None) or {}
        table = {}
        for name, unit, _, _, step in METRICS:
            if name not in self.perf_variables or name in LOWER_IS_BETTER:
                continue
            src = gpu if step == 'gpu' else cpu
            val = _num(src.get(name)) if isinstance(src, dict) else None
            if val is not None and val > 0:
                table[name] = (val, low, None, unit)
        if table:
            self.reference = {self.current_partition.fullname: table}

    @run_before('run')
    def place_job(self):
        self.job.launcher = getlauncher('local')()
        self.job.pin_nodes = [self.node['name']]
        self.job.options += sol.node_job_options(self.node)

    @sanity_function
    def validate(self):
        problems = self.problems()
        if problems:
            raise SanityError('; '.join(problems))
        return True

    def problems(self):
        '''Every problem in the job output, each tagged with its step: "[srun] ..."'''
        n = self.node
        state = ','.join(sorted(set(str(self.job.state or '').split(',')) - {''})) or 'unknown'
        path = os.path.join(self.stagedir, str(self.job.stdout))
        if not os.path.isfile(path):
            return ['[done] node check never ran (%s: No such file or directory; Slurm state %s)'
                    % (self.job.stdout, state)]

        lines = {}
        for tok, fields in sol.read_sol_lines(path):
            lines.setdefault(tok, []).append(fields)

        def first(tok):
            return (lines.get(tok) or [None])[0]

        finished = 'SOL_DONE' in lines     # missing lines only count when the script finished
        skipped = set(f.get('step') for f in lines.get('SOL_SKIP', []))
        builds = dict((f.get('step'), f) for f in lines.get('SOL_BUILD', []))
        cores = _cores(n)
        out = []

        if not finished:
            out.append('[done] node check did not finish (no SOL_DONE; Slurm state %s)' % state)
        elif set(state.split(',')) & BAD_STATES:
            out.append('[done] Slurm job state %s' % state)

        ident = first('SOL_IDENT')
        if ident is None:
            if finished:
                out.append('[ident] no SOL_IDENT line')
        else:
            if ident.get('host') != n['name']:
                out.append('[ident] job ran on %s, expected %s' % (_clean(ident.get('host')), n['name']))
            if ident.get('arch') != n['arch']:
                out.append('[ident] arch is %s, inventory says %s' % (_clean(ident.get('arch')), n['arch']))

        dev = first('SOL_DEVICES')
        expected = _int(n.get('gpu_count')) or 0
        if dev is None:
            if finished:
                out.append('[devices] no SOL_DEVICES line')
        elif _int(dev.get('count')) != expected:
            out.append('[devices] found %s %s devices, Slurm gres says %d'
                       % (_clean(dev.get('count')), n['accel'], expected))

        srun = first('SOL_SRUN')
        if srun is None:
            if finished:
                out.append('[srun] no SOL_SRUN line')
        elif _int(srun.get('tasks')) != cores or _int(srun.get('rc')) != 0:
            out.append('[srun] srun started %s of %d tasks (rc=%s)'
                       % (_clean(srun.get('tasks')), cores, _clean(srun.get('rc'))))
        elif _int(srun.get('hosts')) != 1:
            out.append('[srun] tasks ran on %s hosts, expected 1' % _clean(srun.get('hosts')))

        if 'mpi_hello' not in skipped:
            out += self._mpi_problems(lines, builds.get('mpi_hello'), cores, finished)
        out += self._stream_problems(lines, builds.get('stream'), finished)
        if n['accel'] == 'nvidia' and 'gpu' not in skipped:
            out += self._gpu_problems(lines, builds.get('gpu'), expected, finished)

        seen = []
        for f in lines.get('SOL_TIMEOUT', []):
            msg = '[timeout] %s exceeded %s s' % (_clean(f.get('step')), _clean(f.get('seconds')))
            if msg not in seen:
                seen.append(msg)
        return out + seen

    def _build_problem(self, step, build, finished):
        if build is None:
            return ['[%s] no build result' % step] if finished else []
        if build.get('ok') != '1':
            mod = build.get('module') or 'system'
            return ['[%s] build failed with %s' % (step, 'system gcc' if mod == 'system' else _clean(mod))]
        return []

    def _mpi_problems(self, lines, build, cores, finished):
        out = self._build_problem('mpi_hello', build, finished)
        if out or build is None:
            return out
        hello = lines.get('SOL_MPI_HELLO', [])
        if not hello:
            return ['[mpi_hello] no SOL_MPI_HELLO line: the MPI job did not start or crashed'] if finished else []
        if len(hello) > 1:
            return ['[mpi_hello] %d separate MPI jobs started instead of one with %d ranks '
                    '(PMI wire-up failed?)' % (len(hello), cores)]
        h = hello[0]
        if _int(h.get('ranks')) != cores:
            out.append('[mpi_hello] %s of %d ranks reported' % (_clean(h.get('ranks')), cores))
        if h.get('allreduce_ok') != '1':
            out.append('[mpi_hello] allreduce result wrong')
        if _int(h.get('hosts')) != 1:
            out.append('[mpi_hello] ranks on %s hosts, expected 1' % _clean(h.get('hosts')))
        return out

    def _stream_problems(self, lines, build, finished):
        out = self._build_problem('stream', build, finished)
        if out or build is None:
            return out
        stream = (lines.get('SOL_STREAM') or [None])[0]
        if stream is None:
            return ['[stream] no SOL_STREAM result'] if finished else []
        if stream.get('valid') != '1':
            return ['[stream] validation failed']
        return []

    def _gpu_problems(self, lines, build, expected, finished):
        out = self._build_problem('gpu', build, finished)
        if out or build is None:
            return out
        order, seen = [], {}
        for tok in ('SOL_GPU_INFO', 'SOL_GPU_PCIE', 'SOL_GPU_HBM', 'SOL_GPU_GEMM', 'SOL_GPU_ERROR'):
            for f in lines.get(tok, []):
                uuid = _clean(f.get('uuid') or 'unknown')
                if uuid not in seen:
                    seen[uuid] = {}
                    order.append(uuid)
                seen[uuid].setdefault(tok, []).append(f)
        for uuid in order:
            d = seen[uuid]
            errors = []
            for f in d.get('SOL_GPU_ERROR', []):
                reason = _clean(f.get('reason', ''))
                if reason not in errors:
                    errors.append(reason)
            exits = [e for e in errors if e.startswith('exit ')]
            others = [e for e in errors if not e.startswith('exit ')]
            gemm = d.get('SOL_GPU_GEMM')
            if gemm is None:
                detail = ', '.join(exits)
                if others:
                    detail = (detail + ': ' if detail else '') + ', '.join(others)
                out.append('[gpu] device %s: no GEMM result (%s)' % (uuid, detail or 'no output'))
            else:
                if others:
                    out.append('[gpu] device %s: %s' % (uuid, ', '.join(others)))
                bad = _int(gemm[0].get('errors'))
                if bad != 0:
                    out.append('[gpu] device %s: %s checksum errors in GEMM' % (uuid, _clean(gemm[0].get('errors'))))
            if not errors and finished:
                for tok, what in (('SOL_GPU_INFO', 'info'), ('SOL_GPU_PCIE', 'pcie'), ('SOL_GPU_HBM', 'hbm')):
                    if tok not in d:
                        out.append('[gpu] device %s: no %s result' % (uuid, what))
        reported = sum(1 for uuid in order if 'SOL_GPU_GEMM' in seen[uuid])
        if reported != expected and (finished or reported):
            out.append('[gpu] %d of %d devices reported' % (reported, expected))
        return out
