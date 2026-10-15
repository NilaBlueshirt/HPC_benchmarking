# sol_config.py - ReFrame 4.10 configuration for the Sol pre-release checks.
# One Slurm partition, "compute"; every test picks its own Slurm partition and nodes.
# Module names and limits come from SOL_* variables (site.conf via run_prerelease.sh).

import os
import sys

# append, never insert: ReFrame loads this file inside temp_sys_path(), which pops sys.path by
# position, so an insert(0) here made it drop the last entry (the venv's site-packages: polars)
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tests'))
import sol_common as sol  # noqa: E402

_SUITE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


def _mods(*names):
    return [m for m in names if m]


_mpi_compilers = {'cc': 'mpicc', 'cxx': 'mpicxx', 'ftn': 'mpif90'}
_candidates = sol.mpi_candidates()
_hpcx = dict(_candidates).get('hpcx', '')

_environments = [
    dict(name='node-mpi', modules=_mods(sol.env('MOD_NODE_MPI', 'openmpi/5.0.8-gcc-15.2.0')),
         **_mpi_compilers),
    dict(name='hpcx-cuda', modules=_mods(_hpcx, sol.env('MOD_CUDA', 'cuda-13.2.1-gcc-13.2.0')),
         **_mpi_compilers),
] + [dict(name=key, modules=_mods(mod), **_mpi_compilers) for key, mod in _candidates]

site_configuration = {
    'systems': [{
        'name': 'sol',
        'descr': 'ASU Sol',
        'hostnames': ['.*'],
        'modules_system': os.environ.get('SOL_MODULES_SYSTEM', 'lmod'),
        'partitions': [{
            'name': 'compute',
            'descr': 'Any Sol node; tests add --partition/--nodelist themselves',
            'scheduler': 'slurm',
            'launcher': 'srun',
            'access': [],
            'time_limit': '30m',
            'max_jobs': sol.env_int('MAX_JOBS', 400),
            # every job starts from a clean module environment, whatever the submitting shell had
            'prepare_cmds': [
                'type module >/dev/null 2>&1 || source /etc/profile.d/lmod.sh >/dev/null 2>&1 '
                '|| source /usr/share/lmod/lmod/init/bash >/dev/null 2>&1 || true',
                'module purge >/dev/null 2>&1 || true'
            ],
            'sched_options': {
                # pending reasons that make ReFrame cancel a job; run_prerelease.sh drops ReqNodeNotAvail
                # for the MPI stage, whose jobs wait for each other (RFM_IGNORE_REQNODENOTAVAIL=1)
                'slurm_job_cancel_reasons': [
                    'ReqNodeNotAvail', 'BadConstraints', 'PartitionNodeLimit',
                    'PartitionTimeLimit', 'PartitionConfig', 'InvalidQOS',
                    'InvalidAccount', 'QOSNotAllowed', 'AccountNotAllowed', 'NodeDown',
                    'QOSMaxCpuPerJobLimit', 'QOSMaxNodePerJobLimit', 'QOSMaxWallDurationPerJobLimit',
                    'AssocMaxCpuPerJobLimit', 'AssocMaxNodePerJobLimit',
                    'AssocMaxWallDurationPerJobLimit'
                ],
                'max_sacct_failures': 60,
                'job_submit_timeout': 120
            },
            'environs': ['builtin'] + [e['name'] for e in _environments]
        }]
    }],
    'environments': _environments,
    'storage': [{
        'enable': sol.env('RESULTS_DB', 'yes') != 'no',
        'sqlite_db_file': sol.env('DB_FILE', os.path.join(_SUITE, 'results', 'reframe.db'))
    }],
    'logging': [{
        'perflog_multiline': True,
        'handlers_perflog': [{
            'type': 'filelog',
            'prefix': '%(check_system)s/%(check_partition)s',
            'level': 'info',
            'format': (
                '%(check_job_completion_time)s|%(check_system)s|%(check_partition)s|'
                '%(check_environ)s|%(check_name)s|%(check_node)s|%(check_group)s|'
                '%(check_mpi)s|%(check_bench)s|%(check_job_nodelist)s|%(check_jobid)s|'
                '%(check_result)s|%(check_perf_var)s|%(check_perf_value)s|'
                '%(check_perf_unit)s|%(check_perf_ref)s|%(check_perf_lower_thres)s|'
                '%(check_perf_upper_thres)s|%(check_perf_result)s'
            ),
            'datefmt': '%FT%T%:z',
            'append': True
        }]
    }]
}
