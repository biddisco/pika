#!/usr/bin/env python
# coding: utf-8

build_directory="@PROJECT_BINARY_DIR@"
binary_directory="@CMAKE_RUNTIME_OUTPUT_DIRECTORY@"
try:
    mpi_modes = int("@PIKA_MPI_MODES_LOOP_COUNT@")
except ValueError:
    mpi_modes = 63

# ------------------------------------------------------------------
# system imports
# ------------------------------------------------------------------
import os, sys
import argparse
from itertools import product
import functools
import operator

# ------------------------------------------------------------------
# local imports
# ------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from dlaf repo
import miniapps as mp
import systems

# ------------------------------------------------------------------
# system defs: extend daint rules to allow other ranks per node
# ------------------------------------------------------------------
systems.cscs["daint-mc"]["Allowed rpns"] = [1, 2, 4]

# ------------------------------------------------------------------
# system defs: jb laptop - mpich
# ------------------------------------------------------------------
systems.cscs["oryx2-mpich"] = {
    "Machine": "system76",
    "Cores": 16,
    "Threads per core": 1,
    "Allowed rpns": [4],
    "Multiple rpn in same job": True,
    "GPU": True,
    # cd {job_path} ;
    "Launch command": "cd {job_path} ; source {job_file}",
    "Run command": "mpiexec -n {total_ranks} -bind-to hwthread:PE={threads_per_rank*total_ranks} ",
    "Batch preamble": """
#!/bin/bash -l
spack load mpich
# Commands
""",
}

# ------------------------------------------------------------------
# system defs: jb laptop - openmpi
# ------------------------------------------------------------------
systems.cscs["oryx2-openmpi"] = systems.cscs["oryx2-mpich"].copy()
systems.cscs["oryx2-openmpi"]["Run command"]= "mpirun -n {total_ranks} --map-by socket:PE={threads_per_rank} --bind-to core "
systems.cscs["oryx2-openmpi"]["Batch preamble"]= """
#!/bin/bash -l
spack load openmpi /mewprpp
# Commands
"""

# ------------------------------------------------------------------
# command generation for mpi_ring_async_sender_receiver
# ------------------------------------------------------------------
def mpi_ring_async_sender_receiver(
    system,
    lib,
    miniapp_dir,
    nodes,
    nruns,
    rpn,
    iter,
    rounds,
    flight,
    nbytes,
    mpi_mode,
    suffix="na",
    extra_flags="",
    dtype=" ",
    env="",
):
    # add mpi_mode, rpn, nodes to extra flags
    new_flags = f"--pp-info \"mpi_completion, {mpi_mode}, rpn, {rpn}, nodes, {nodes}, " 
    extra_flags = extra_flags.replace("--pp-info \"", new_flags)
    #
    [total_ranks, cores_per_rank, threads_per_rank] = mp._computeResourcesNeededList(system, nodes, rpn)
    # print(f"total_ranks: {total_ranks}, cores_per_rank: {cores_per_rank}, threads_per_rank: {threads_per_rank}")
    app = f"{miniapp_dir}/mpi_ring_async_sender_receiver_test"
    n_iter = iter[nbytes] 
    opts = f"{prog_args} --standalone --iterations {n_iter} --rounds {rounds} --in-flight-limit {flight} --message-bytes {nbytes} --pika:mpi-completion-mode {mpi_mode} {extra_flags}"

    mp._checkAppExec(app)
    cmd = f"{app} {opts}".strip() + f" >> mpi_ring_{mpi_mode}.out 2>&1"
    return cmd, env.strip()

# ------------------------------------------------------------------
# Utility function to handle comma separated args as list
# ------------------------------------------------------------------
class SplitArgs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values.split(","))
        print(f"set {self.dest} to {values.split(',')}")

# ------------------------------------------------------------------
# Command line params
# ------------------------------------------------------------------
def get_command_line_args(notebook_args=None):
    parser = argparse.ArgumentParser(description="Generator for miniapp benchmarks")
    parser.add_argument(
        "-m",
        "--machine",
        default="",
        action="store",
        help="select machine batch job config/preamble",
    )
    parser.add_argument(
        "-b",
        "--backend",
        default="",
        action="store",
        help="specify the backend used (mc/gpu)",
    )
    parser.add_argument(
        "-d",
        "--binary_dir",
        default=binary_directory,
        action="store",
        help="directory where the binaries were built/stored - usually $build_dir/bin)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=cwd,
        action="store",
        help="base directory to generate job scripts in",
    )
    parser.add_argument(
        "-t", "--timeout", 
        default=10, 
        action="store", 
        help="nominal executable timeout period multiplier in seconds (timeout*ranks is used as the actual timeout)",
    )
    parser.add_argument(
        "-p",
        "--prog-args",
        default=[],
        action=SplitArgs,
        help="list of args that are added to the executable invocation",
    )
    parser.add_argument(
        "-n",
        "--nodes",
        default=[1,2,4],
        action=SplitArgs,
        help="list number of nodes [1,2,4,8,...] to generate job scripts for",
    )
    parser.add_argument(
        "-r",
        "--branchname",
        default="develop",
        action="store",
        help="branch name to display in the job result",
    )
    # add debug flag argument
    parser.add_argument(
        "-g",
        "--debug",
        action="store_true",
        help="use debug flag in generation script",
    )
    parser.add_argument(
        "-a",
        "--miniapps",
        default=["mpi_ring"],
        action=SplitArgs,
        help="list of miniapp names to run",
    )
    parser.add_argument(
        "-u",
        "--uenv",
        default=None,
        action="store",
        help="uenv squashfs image to load for alps machines",
    )
    parser.add_argument(
        "-w",
        "--wrapper",
        default=f"{build_directory}/scripts",
        help="path to default wrapper location for helper scripts such as 'ompi-wrapper' or 'gpu2ranks_slurm_cuda'",
    )
    parser.add_argument(
        "-v",
        "--reservation",
        default="",
        help="reservation name for sbatch launches if needed",
    )
    return parser.parse_args()

# ------------------------------------------------------------------
# parse command line args
# getting the machine name is necessary to determine hardware specific constraints
# ------------------------------------------------------------------
cwd = os.getcwd()
args = get_command_line_args()
if args.machine == "":
    machine_names = list(systems.cscs.keys())
    sys.exit(f"please supply machine name - one of {machine_names}")

os.makedirs(args.output_dir, exist_ok=True)

# systems.cscs['eiger']["Allowed rpns"] = [1]

# ------------------------------------------------------------------
system          = systems.cscs[args.machine]
backend         = args.backend
prog_args       = " ".join(args.prog_args)
nodes_arr       = [int(node) for node in args.nodes]
poll_arr        = [16]
inflight_arr    = list(reversed([1,2,4,8,16,32,64,128]))
inflight_arr    = [32]
bytes_arr       = [64,       131072]
iter_arr        = {64: 5000, 131072: 1000}
rounds_arr      = [5]
threads_arr     = [16]
mode_arr        = list(range(0, mpi_modes+1)) 
sched_arr       = ["local-priority"]
rpn_arr         = system["Allowed rpns"]
repetitions     = 10
miniapps        = args.miniapps

print("Machine              :", args.machine)
print("timeout (per test)   :", args.timeout)
print("mpi_modes            :", mpi_modes)
print("backend              :", backend)
print("ranks per node       :", rpn_arr)
print("poll sizes           :", poll_arr)
print("flight_arr sizes     :", inflight_arr)
print("completion mode      :", mode_arr)
print("scheduler            :", sched_arr)
print("nodes                :", nodes_arr)
print("iterations           :", iter_arr)
print("rounds               :", rounds_arr)
print("bytes                :", bytes_arr)
print("threads              :", threads_arr)
print("test_apps            :", miniapps)

# ------------------------------------------------------------------
def extra_csv_named(poll, sched, machine, branch, name):
    return f'--csv --pp-info "mpi_poll, {poll}, scheduler, {sched}, machine, {machine}, branch, {branch}, benchmark, {name}"'

# ------------------------------------------------------------------
# if an executable wrapper script is needed, we must set the path here
systems.cscs[args.machine]["Run command"] = "timeout {timeout} " + systems.cscs[args.machine]["Run command"].replace("{wrapper}", args.wrapper)
original_run_command = systems.cscs[args.machine]["Run command"]
systems.cscs[args.machine]["Timeout per rank"] = int(args.timeout)
print(args.machine, "Run command:", original_run_command)
 
# ------------------------------------------------------------------
# on alps machines we may require the setting of a uenv 
if "{uenv}" in systems.cscs[args.machine]["Batch preamble"]:
    if args.uenv is None:
        sys.exit(f"please supply uenv path for {args.machine}")
    systems.cscs[args.machine]["Batch preamble"] = systems.cscs[args.machine]["Batch preamble"].replace("{uenv}", args.uenv)
srun_args = ""

# ------------------------------------------------------------------
# on slurm machines we might need to set a reservation
if "sbatch" in systems.cscs[args.machine]["Launch command"] and args.reservation != "":
    systems.cscs[args.machine]["Launch command"] = systems.cscs[args.machine]["Launch command"].replace("sbatch", f"sbatch --reservation={args.reservation}")
    print(args.machine, "Launch command:", systems.cscs[args.machine]["Launch command"])

# job limit is usually 12|24 hours, clamp to 1 minute less
job_limit = 12
max_walltime = job_limit*60 - 1  

branch = args.branchname

# ------------------------------------------------------------------
# mpi-ring-send
# ------------------------------------------------------------------
if "mpi_ring" in miniapps:
    for duplicate in range(0, repetitions):
        for nnodes in nodes_arr:
            num_runs = repetitions * functools.reduce(operator.mul, map(len, [mode_arr, poll_arr, sched_arr, rpn_arr, iter_arr, rounds_arr, inflight_arr, bytes_arr]), 1)
            total_seconds = args.timeout * num_runs * nnodes
            total_minutes = total_seconds // 60
            if total_minutes > max_walltime:
                print(f"Total time: {total_minutes} exceeds job limit, reducing job time")
                total_minutes = min(total_minutes, max_walltime) 
            print(f"Total time: {total_minutes}", num_runs)
            run = mp.StrongScaling(system, "mpi_ring_test", f"job_pika_{nnodes}_{duplicate}", [nnodes], total_minutes)

            for poll, sched  in product(poll_arr, sched_arr):
                evar = f"PIKA_MPI_POLL_SIZE={poll}"
                run.add(
                    miniapp=mpi_ring_async_sender_receiver,
                    lib="mpi",
                    miniapp_dir=args.binary_dir,
                    params={"rpn": rpn_arr, "iter": iter_arr, "rounds": rounds_arr, "flight": inflight_arr, "nbytes": bytes_arr, "mpi_mode": mode_arr},
                    nruns=1,                    
                    suffix=f"_{duplicate}",
                    extra_flags=extra_csv_named(
                        poll,
                        sched,
                        args.machine,
                        branch,
                        "mpi_ring_send",
                    ),
                    env=evar,
                    srun_args=srun_args,
                )
            run.submit(f"{args.output_dir}/{duplicate}", args.debug)

# ------------------------------------------------------------------
# DLA-Future
# ------------------------------------------------------------------
else:
    size_arr = [40960]
    block_arr = list(reversed([1024]))
    nruns = 5
    dlafpath = args.binary_dir

    all_miniapps = [
        "gen2std",
        "evp",
        "bt_red2band",
        "gevp",
        "trid_evp",
        "band2trid",
        "red2band",
        "cholesky",
        "bt_band2trid",
        "trsm",
    ]

    # check miniapps we are executing
    miniapps = args.miniapps
    for miniapp in miniapps:
        if not miniapp in all_miniapps:
            print(f"{miniapp} is not a valid miniapp")
            raise ValueError        

    for nranks in nodes_arr:

        num_runs = nruns * functools.reduce(
            operator.mul,
            map(len, [mode_arr, poll_arr, sched_arr, inflight_arr, rpn_arr, block_arr, size_arr]),
            1,
        )
        total_seconds = 60 * num_runs * nranks
        total_minutes = total_seconds // 60
        total_minutes = min(total_minutes, max_walltime)
        print(f"Total time: {total_minutes}", num_runs)
        run = mp.StrongScaling(
            system, "dlaf_test", f"job_dlaf_{nranks}", [nranks], total_minutes
        )

        for mode in mode_arr:
            for poll, sched in product(poll_arr, sched_arr):
                evar = f"PIKA_MPI_COMPLETION_MODE={mode} PIKA_MPI_POLL_SIZE={poll}"

                for sched in sched_arr:
                    evar = f"PIKA_MPI_COMPLETION_MODE={mode} PIKA_MPI_POLL_SIZE={poll}"

                    if "cholesky" in miniapps:
                        run.add(
                            mp.chol,
                            "dlaf",
                            dlafpath,
                            {"rpn": rpn_arr, "m_sz": size_arr, "mb_sz": block_arr},
                            nruns,
                            extra_flags=extra_csv_named(
                                poll,
                                sched,
                                args.machine,
                                branch,
                                "cholesky",
                            ),
                            env=evar,
                            srun_args=srun_args,
                        )
                    if "gen2std" in miniapps:
                        run.add(
                            mp.gen2std,
                            "dlaf",
                            dlafpath,
                            {"rpn": rpn_arr, "m_sz": size_arr, "mb_sz": block_arr},
                            nruns,
                            extra_flags=extra_csv_named(
                                poll,
                                sched,
                                args.machine,
                                branch,
                                "gen2std",
                            ),
                            env=evar,
                            srun_args=srun_args,
                        )
                    if "red2band" in miniapps:
                        run.add(
                            mp.red2band,
                            "dlaf",
                            dlafpath,
                            {
                                "rpn": rpn_arr,
                                "m_sz": size_arr,
                                "mb_sz": block_arr,
                                "band": 128,
                            },
                            nruns,
                            extra_flags=extra_csv_named(
                                poll,
                                sched,
                                args.machine,
                                branch,
                                "red2band",
                            ),
                            env=evar,
                            srun_args=srun_args,
                        )
                    if "band2trid" in miniapps:
                        run.add(
                            mp.band2trid,
                            "dlaf",
                            dlafpath,
                            {
                                "rpn": rpn_arr,
                                "m_sz": size_arr,
                                "mb_sz": block_arr,
                                "band": 128,
                            },
                            nruns,
                            extra_flags=extra_csv_named(
                                poll,
                                sched,
                                args.machine,
                                branch,
                                "band2trid",
                            ),
                            env=evar,
                            srun_args=srun_args,
                        )
                    if "trid_evp" in miniapps:
                        run.add(
                            mp.trid_evp,
                            "dlaf",
                            dlafpath,
                            {"rpn": rpn_arr, "m_sz": size_arr, "mb_sz": block_arr},
                            nruns,
                            extra_flags=extra_csv_named(
                                poll,
                                sched,
                                args.machine,
                                branch,
                                "trid_evp",
                            ),
                            env=evar,
                            srun_args=srun_args,
                        )
                    if "bt_band2trid" in miniapps:
                        run.add(
                            mp.bt_band2trid,
                            "dlaf",
                            dlafpath,
                            {
                                "rpn": rpn_arr,
                                "m_sz": size_arr,
                                "mb_sz": block_arr,
                                "band": 128,
                                "n_sz": None,
                            },
                            nruns,
                            extra_flags=extra_csv_named(
                                poll,
                                sched,
                                args.machine,
                                branch,
                                "bt_band2trid",
                            ),
                            env=evar,
                            srun_args=srun_args,
                        )
                    if "bt_red2band" in miniapps:
                        run.add(
                            mp.bt_red2band,
                            "dlaf",
                            dlafpath,
                            {
                                "rpn": rpn_arr,
                                "m_sz": size_arr,
                                "mb_sz": block_arr,
                                "band": 128,
                                "n_sz": None,
                            },
                            nruns,
                            extra_flags=extra_csv_named(
                                poll,
                                sched,
                                args.machine,
                                branch,
                                "bt_red2band",
                            ),
                            env=evar,
                            srun_args=srun_args,
                        )
                    if "trsm" in miniapps:
                        run.add(
                            mp.trsm,
                            "dlaf",
                            dlafpath,
                            {
                                "rpn": rpn_arr,
                                "m_sz": size_arr,
                                "mb_sz": block_arr,
                                "n_sz": None,
                            },
                            nruns,
                            extra_flags=extra_csv_named(
                                poll,
                                sched,
                                args.machine,
                                branch,
                                "trsm",
                            ),
                            env=evar,
                            srun_args=srun_args,
                        )
                    if "evp" in miniapps:
                        run.add(
                            mp.evp,
                            "dlaf",
                            dlafpath,
                            {
                                "rpn": rpn_arr,
                                "m_sz": size_arr,
                                "mb_sz": block_arr,
                                "min_band": None,
                            },
                            nruns,
                            extra_flags=extra_csv_named(
                                poll,
                                sched,
                                args.machine,
                                branch,
                                "evp",
                            ),
                            env=evar,
                            srun_args=srun_args,
                        )
                    if "gevptrsm" in miniapps:
                        run.add(
                            mp.gevp,
                            "dlaf",
                            dlafpath,
                            {
                                "rpn": rpn_arr,
                                "m_sz": size_arr,
                                "mb_sz": block_arr,
                                "min_band": None,
                            },
                            nruns,
                            extra_flags=extra_csv_named(
                                poll,
                                sched,
                                args.machine,
                                branch,
                                "gevp",
                            ),
                            env=evar,
                            srun_args=srun_args,
                        )
        run.submit(args.output_dir, args.debug)
