# mpi_checks.py - multi-node MPI tests of the Sol pre-release checks (the "mpi" stage).
# SolPairwiseBW measures every node pair of an MPI group with pairbw.c (node-mpi environ).
# SolOsuPt2Pt and SolOsuCollective run OSU micro-benchmarks for each group x MPI candidate;
# SolOsuGpuDirect runs OSU bw/latency between GPU buffers (hpcx-cuda), only with SOL_GPU_DIRECT=yes.
# Groups come from SOL_MPI_GROUPS. A group with fewer than 2 usable nodes (testable in nodes.json
# and not listed in SOL_SKIP_NODES_FILE) is left out.
# All jobs of a group share its first nodes and run one after another: each job may run for
# SOL_MPI_TIME_LIMIT and is cancelled only after SOL_MPI_MAX_PENDING in the queue.
# srun gets --mpi=SOL_SRUN_MPI_<key> (key = the MPI candidate, hpcx for GPU-direct, node_mpi for
# pairwise), else SOL_SRUN_MPI, else pmix.
#
# Usage: reframe -C config/sol_config.py -c tests/mpi_checks.py -r   (run_prerelease.sh does this)

import os
import re
import shlex
import statistics
import sys

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.builtins import (fixture, parameter, performance_function, run_after,
                                   run_before, sanity_function)
from reframe.core.exceptions import SanityError

sys.path.append(os.path.dirname(os.path.abspath(__file__)))   # append: see config/sol_config.py
import sol_common as sol  # noqa: E402

OSU_URL = 'https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.5.2.tar.gz'

# The pending limit covers a job's wait behind the other jobs of its group, not just one job.
MPI_TIME_LIMIT = sol.env('MPI_TIME_LIMIT', '15m')
MPI_MAX_PENDING = sol.env('MPI_MAX_PENDING', '4h')

# bench: (binary, options, largest message size, {perf variable: (message size, unit)})
OSU_BENCH = {
    'bw': ('osu_bw', ['-m', '1:4194304'], 4194304,
           {'bw_4M': (4194304, 'MB/s'), 'bw_64K': (65536, 'MB/s')}),
    'latency': ('osu_latency', ['-m', '1:4194304'], 4194304,
                {'lat_8B': (8, 'us'), 'lat_64K': (65536, 'us')}),
    'allreduce': ('osu_allreduce', ['-m', '8:1048576', '-i', '200', '-x', '20'], 1048576,
                  {'lat_8B': (8, 'us'), 'lat_1M': (1048576, 'us')}),
    'alltoall': ('osu_alltoall', ['-m', '1:4096', '-i', '100', '-x', '10'], 4096,
                 {'lat_8B': (8, 'us'), 'lat_4K': (4096, 'us')}),
}


def usable_groups(inventory=None, skip=None):
    '''MPI groups with at least 2 usable nodes, as parameter values (plain dicts).'''
    inv = inventory if inventory is not None else sol.load_inventory()
    skip = sol.skip_nodes() if skip is None else skip
    cores = dict((n['name'], int(n.get('cores') or 0)) for n in inv.get('nodes', []))
    groups = []
    for g in sol.mpi_groups():
        nodes = sol.usable_group_nodes(g, inventory=inv, skip=skip)
        if len(nodes) >= 2:
            groups.append({'name': g['name'], 'partition': g['partition'], 'gpu': g['gpu'],
                           'coll_nodes': g['coll_nodes'], 'nodes': nodes,
                           'cores': [cores[h] for h in nodes]})
    return groups


def collective_layout(group):
    '''(nodes, tasks per node) for a collective: the first min(coll_nodes, usable) nodes
    (coll_nodes below 2 counts as 2), as many ranks per node as the smallest of them has cores.'''
    n = min(max(group['coll_nodes'], 2), len(group['nodes']))
    return group['nodes'][:n], max(1, min(group['cores'][:n]))


GROUPS = usable_groups()
GPU_GROUPS = [g for g in GROUPS if g['gpu']]
MPI_KEYS = [key for key, _ in sol.mpi_candidates()]


def osu_rows(text):
    '''(size, value) rows of the last OSU table: "# Size ..." header, then "<size> <value> ...".'''
    rows, in_table = [], False
    for line in text.splitlines():
        f = line.split()
        if line.startswith('# Size'):
            rows, in_table = [], True
        elif in_table and len(f) >= 2 and f[0].isdigit():
            try:
                rows.append((int(f[0]), float(f[1])))
            except ValueError:
                rows.append((int(f[0]), float('nan')))
    return rows


def osu_problems(text, max_size, binary):
    '''Sanity messages for one OSU run; [] when the table is complete.'''
    if not re.search(r'^# Size', text, re.M):
        return ['no OSU result table from %s' % binary]
    rows = osu_rows(text)
    if not rows:
        return ['OSU table from %s has no rows' % binary]
    msgs = []
    if rows[-1][0] != max_size:
        msgs.append('OSU table stops at %d bytes, expected %d' % (rows[-1][0], max_size))
    bad = [str(s) for s, v in rows if not v > 0]
    if bad:
        msgs.append('OSU value not positive at size(s) %s' % ','.join(bad[:8]))
    return msgs


def pairbw_problems(lines, n):
    '''Sanity messages for pairbw output [(token, fields), ...] of an n-node run.'''
    expected = n * (n - 1) // 2
    summary = [f for t, f in lines if t == 'SOL_PAIRBW']
    pairs = [f for t, f in lines if t == 'SOL_PAIR']
    if not summary:
        return ['pairbw did not finish: no SOL_PAIRBW line, %d of %d pairs reported'
                % (len(pairs), expected)]
    s = summary[-1]
    if len(summary) > 1 or s.get('nodes') != str(n):
        return ['pairbw started as %d job(s) of %s rank(s), expected one job of %d ranks '
                '(MPI start-up problem)' % (len(summary), s.get('nodes'), n)]
    msgs = []
    if not s.get('pairs') == s.get('expected') == str(expected):
        msgs.append('pairbw measured pairs=%s expected=%s, should be %d'
                    % (s.get('pairs'), s.get('expected'), expected))
    if len(pairs) != expected:
        msgs.append('%d SOL_PAIR lines, expected %d' % (len(pairs), expected))
    nobw = ['%s-%s' % (p.get('a'), p.get('b')) for p in pairs if not _num(p.get('bibw_mbs')) > 0]
    if nobw:
        msgs.append('%d pair(s) without bandwidth: %s' % (len(nobw), ' '.join(nobw[:5])))
    hosts = set(p.get('a') for p in pairs) | set(p.get('b') for p in pairs)
    links = set(frozenset((p.get('a'), p.get('b'))) for p in pairs if p.get('a') != p.get('b'))
    if pairs and (len(hosts) != n or len(links) != len(pairs)):
        msgs.append('pairs cover %d distinct hosts and %d distinct links, expected %d and %d '
                    '(ranks sharing a node?)' % (len(hosts), len(links), n, len(pairs)))
    return msgs


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


def _read(path):
    try:
        with open(path, errors='replace') as fp:
            return fp.read()
    except OSError:
        return ''


def stderr_hint(text):
    '''The most telling line of a job's stderr, for a sanity message ("" if empty).'''
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        if re.search(r'error|abort|fail|not found|cannot|unable|requires|killed|signal', ln, re.I):
            return ln[:200]
    return lines[0][:200] if lines else ''


def never_started(test):
    '''Message when the job left no stdout at all (cancelled while pending), else "".'''
    if os.path.exists(os.path.join(test.stagedir, test.job.stdout)):
        return ''
    return ('%s: No such file: the job never started (Slurm state %s; pending longer than '
            'MPI_MAX_PENDING=%s?)' % (test.job.stdout, test.job.state, MPI_MAX_PENDING))


def job_problems(test):
    '''Messages about the job itself: time limit hit, plus the stderr hint.'''
    msgs = []
    if test.job.state == 'TIMEOUT':
        msgs.append('job hit its time limit (Slurm state TIMEOUT)')
    hint = stderr_hint(_read(os.path.join(test.stagedir, test.job.stderr)))
    if hint:
        msgs.append('stderr: %s' % hint)
    return msgs


def srun_mpi(key):
    '''srun --mpi plugin for one MPI: SRUN_MPI_<key>, else SRUN_MPI, else pmix.'''
    return sol.env('SRUN_MPI_' + key, sol.env('SRUN_MPI', 'pmix'))


def setup_mpi_job(test, nodes, partition, mpi_key, extra_options=()):
    '''Pin the job to its nodes, add the Slurm options and the srun --mpi plugin of mpi_key.'''
    test.job.pin_nodes = list(nodes)
    test.job.options += sol.common_job_options(partition) + list(extra_options)
    if test.job.launcher.registered_name == 'srun':
        test.job.launcher.options = ['--mpi=' + srun_mpi(mpi_key)]


@rfm.simple_test
class SolPairwiseBW(rfm.RegressionTest):
    '''All-pairs bidirectional bandwidth and latency inside one MPI group, one rank per node.'''

    descr = 'pairbw: every node pair of an MPI group'
    group = parameter(GROUPS, fmt=lambda g: g['name'])
    valid_systems = ['sol:compute']
    valid_prog_environs = ['node-mpi']
    sourcesdir = 'src'
    sourcepath = 'pairbw.c'
    build_system = 'SingleSource'
    build_locally = True
    # n(n-1)/2 pairs, bandwidth one pair at a time: about 3 minutes for 107 nodes at 200 Gb/s
    executable_opts = ['16777216', '20', '1000']
    num_tasks_per_node = 1
    exclusive_access = True
    time_limit = MPI_TIME_LIMIT
    max_pending_time = MPI_MAX_PENDING
    tags = {'mpi', 'pairwise', 'node-mpi'}

    @run_after('init')
    def set_layout(self):
        self.num_tasks = len(self.group['nodes'])
        self.tags |= {self.group['name']}

    @run_before('compile')
    def set_cflags(self):
        self.build_system.cflags = ['-O2']

    @run_before('run')
    def set_job(self):
        setup_mpi_job(self, self.group['nodes'], self.group['partition'], 'node_mpi')

    def _pairs(self):
        lines = sol.read_sol_lines(os.path.join(self.stagedir, self.job.stdout))
        return [f for t, f in lines if t == 'SOL_PAIR']

    @sanity_function
    def validate(self):
        missing = never_started(self)
        if missing:
            raise SanityError(missing)
        lines = sol.read_sol_lines(os.path.join(self.stagedir, self.job.stdout))
        msgs = pairbw_problems(lines, len(self.group['nodes']))
        if msgs:
            raise SanityError('; '.join(msgs + job_problems(self)))
        return True

    @performance_function('MB/s')
    def link_bw_min(self):
        return min(_num(p.get('bibw_mbs')) for p in self._pairs())

    @performance_function('MB/s')
    def link_bw_median(self):
        return round(statistics.median(_num(p.get('bibw_mbs')) for p in self._pairs()), 2)

    @performance_function('us')
    def lat_max(self):
        return max(_num(p.get('lat_us')) for p in self._pairs())


class SolOsuBuild(rfm.CompileOnlyRegressionTest):
    '''Fixture: OSU micro-benchmarks built on the login node with the environ's MPI.'''

    descr = 'OSU micro-benchmarks build'
    build_system = 'CustomBuild'
    build_locally = True
    sourcesdir = None
    needed = ('osu_bw', 'osu_latency', 'osu_allreduce', 'osu_alltoall')

    def configure_cmds(self, prefix):
        return ['./configure CC=mpicc CXX=mpicxx --prefix=%s' % shlex.quote(prefix)]

    @run_before('compile')
    def set_commands(self):
        tarball = sol.env('OSU_TARBALL')
        if tarball:
            fetch = 'cp %s osu.tar.gz' % shlex.quote(os.path.abspath(tarball))
        else:
            fetch = 'curl -fsSL -o osu.tar.gz %s' % shlex.quote(sol.env('OSU_URL', OSU_URL))
        self.build_system.commands = (
            [fetch, 'mkdir -p osu-src', 'tar xzf osu.tar.gz -C osu-src --strip-components=1',
             'cd osu-src']
            + self.configure_cmds(os.path.join(self.stagedir, 'osu-install'))
            + ['make -j 8', 'make install'])

    @sanity_function
    def validate(self):
        root = os.path.join(self.stagedir, 'osu-install')
        found = set()
        for d, _, files in os.walk(root):
            found.update(f for f in files if os.access(os.path.join(d, f), os.X_OK))
        missing = [b for b in self.needed if b not in found]
        if missing:
            raise SanityError('OSU build: %s not found under %s' % (', '.join(missing), root))
        return True


class SolOsuBuildCuda(SolOsuBuild):
    '''Fixture: OSU micro-benchmarks with CUDA buffer support (hpcx-cuda environ).'''

    descr = 'OSU micro-benchmarks build with CUDA'
    needed = ('osu_bw', 'osu_latency')

    def configure_cmds(self, prefix):
        # CUDA comes from the module's nvcc. The stubs directory lets a login node without the
        # NVIDIA driver link -lcuda; the driver's libcuda.so.1 is used at run time.
        return ['NVCC_BIN=$(command -v nvcc)',
                'CUDA_DIR=$(dirname "$(dirname "$(readlink -f "$NVCC_BIN")")")',
                './configure CC=mpicc CXX=mpicxx --prefix=%s --enable-cuda --with-cuda="$CUDA_DIR" '
                'LDFLAGS="-L$CUDA_DIR/lib64/stubs"' % shlex.quote(prefix)]


class SolOsuBase(rfm.RunOnlyRegressionTest):
    '''Common part of the OSU tests: job layout, binary lookup, sanity and perf.'''

    valid_systems = ['sol:compute']
    sourcesdir = None
    exclusive_access = True
    time_limit = MPI_TIME_LIMIT
    max_pending_time = MPI_MAX_PENDING
    executable = '"$OSU_BIN"'
    tags = {'mpi', 'osu'}

    # set by the subclasses in a post-init hook
    job_nodes = []
    osu_opts = []
    extra_job_options = []
    mpi_key = ''           # picks SRUN_MPI_<mpi_key>

    @run_before('run')
    def set_job(self):
        binary = OSU_BENCH[self.bench][0]
        osu_root = os.path.join(self.osu_build.stagedir, 'osu-install')
        self.env_vars['OSU_ROOT'] = shlex.quote(osu_root)
        self.prerun_cmds = [
            'OSU_BIN=$(find "$OSU_ROOT" -name %s -type f -perm -u+x | head -1)' % binary,
            '[ -n "$OSU_BIN" ] || echo "%s not found under $OSU_ROOT" >&2' % binary]
        self.executable_opts = list(self.osu_opts)
        setup_mpi_job(self, self.job_nodes, self.group['partition'], self.mpi_key,
                      self.extra_job_options)

    @sanity_function
    def validate(self):
        missing = never_started(self)
        if missing:
            raise SanityError(missing)
        binary, _, max_size, _ = OSU_BENCH[self.bench]
        msgs = osu_problems(_read(os.path.join(self.stagedir, self.job.stdout)), max_size, binary)
        if msgs:
            raise SanityError('; '.join(msgs + job_problems(self)))
        return True

    @run_before('performance')
    def set_perf_variables(self):
        self.perf_variables = dict(
            (name, sn.make_performance_function(
                sn.extractsingle(r'^%d[ \t]+(\S+)' % size, self.stdout, 1, float), unit))
            for name, (size, unit) in OSU_BENCH[self.bench][3].items())


@rfm.simple_test
class SolOsuPt2Pt(SolOsuBase):
    '''OSU bandwidth or latency between the first two usable nodes of a group.'''

    descr = 'OSU point-to-point, 2 nodes, 1 rank per node'
    group = parameter(GROUPS, fmt=lambda g: g['name'])
    mpi = parameter(MPI_KEYS)
    bench = parameter(['bw', 'latency'])
    osu_build = fixture(SolOsuBuild, scope='environment')
    num_tasks = 2
    num_tasks_per_node = 1

    @run_after('init')
    def set_layout(self):
        self.valid_prog_environs = [self.mpi]
        self.mpi_key = self.mpi
        self.job_nodes = self.group['nodes'][:2]
        self.osu_opts = OSU_BENCH[self.bench][1]
        self.tags |= {self.group['name'], self.mpi}


@rfm.simple_test
class SolOsuCollective(SolOsuBase):
    '''OSU allreduce or alltoall latency on up to coll_nodes nodes of a group, all cores.'''

    descr = 'OSU collective, up to coll_nodes nodes, one rank per core'
    group = parameter(GROUPS, fmt=lambda g: g['name'])
    mpi = parameter(MPI_KEYS)
    bench = parameter(['allreduce', 'alltoall'])
    osu_build = fixture(SolOsuBuild, scope='environment')

    @run_after('init')
    def set_layout(self):
        self.valid_prog_environs = [self.mpi]
        self.mpi_key = self.mpi
        self.job_nodes, self.num_tasks_per_node = collective_layout(self.group)
        self.num_tasks = len(self.job_nodes) * self.num_tasks_per_node
        self.osu_opts = OSU_BENCH[self.bench][1]
        self.tags |= {self.group['name'], self.mpi}


class SolOsuGpuDirect(SolOsuBase):
    '''OSU bandwidth or latency between GPU buffers on two nodes of a GPU group (hpcx-cuda).'''

    descr = 'OSU point-to-point between GPU buffers, 2 nodes, 1 GPU each'
    group = parameter(GPU_GROUPS, fmt=lambda g: g['name'])
    bench = parameter(['bw', 'latency'])
    osu_build = fixture(SolOsuBuildCuda, scope='environment')
    valid_prog_environs = ['hpcx-cuda']
    num_tasks = 2
    num_tasks_per_node = 1
    extra_job_options = ['--gres=gpu:1']
    mpi_key = 'hpcx'

    @run_after('init')
    def set_layout(self):
        self.job_nodes = self.group['nodes'][:2]
        self.osu_opts = ['-d', 'cuda'] + OSU_BENCH[self.bench][1] + ['D', 'D']
        self.tags |= {self.group['name'], 'hpcx-cuda', 'gpudirect'}


if sol.env('GPU_DIRECT', 'no').lower() == 'yes':
    rfm.simple_test(SolOsuGpuDirect)
