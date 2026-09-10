#!/usr/bin/env python3
"""
Run a file-splittable command as N parallel jobs and hadd the outputs.

The command must accept --njobs/--jobnr (partitioning its input files as
inputfiles[jobnr::njobs], as applyRegFast.py does) and -o for its output
file.  This script appends "--njobs N --jobnr I -o <part>" to the command
for each job, runs the jobs in parallel, and on success merges the part
files with hadd into the final output.

Example:
  runSplitJobs.py --njobs 8 -o out.root -- \\
      python3 Scouting/Tools/scripts/applyRegFast.py -i in1.root in2.root ... --jpsi-sel

Per-job stdout/stderr goes to <output>_jobI.log next to the output file.
Parts and logs are deleted after a successful merge unless --keep-parts.
"""
import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
import time


def job_paths(outputfile, jobnr):
    base, ext = os.path.splitext(outputfile)
    return f"{base}_job{jobnr}{ext}", f"{base}_job{jobnr}.log"


def run_job(command, njobs, jobnr, outputfile):
    partfile, logfile = job_paths(outputfile, jobnr)
    cmd = command + ["--njobs", str(njobs), "--jobnr", str(jobnr), "-o", partfile]
    start = time.time()
    with open(logfile, "w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    status = "done" if result.returncode == 0 else f"FAILED (exit {result.returncode}, see {logfile})"
    print(f"job {jobnr}: {status} in {time.time() - start:.0f}s")
    return result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--njobs", type=int, required=True, help="Number of jobs to split into")
    parser.add_argument("--max-parallel", type=int, default=None, help="Max jobs running at once (default: njobs)")
    parser.add_argument("-o", "--outputfile", required=True, help="Final merged output file")
    parser.add_argument("--keep-parts", action="store_true", help="Keep the per-job outputs and logs after merging")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run (prefix with --)")
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("no command given (put it after --)")
    if shutil.which("hadd") is None:
        parser.error("hadd not found in PATH (set up the CMSSW/ROOT environment first)")

    max_parallel = args.max_parallel or args.njobs
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_parallel) as pool:
        returncodes = list(
            pool.map(lambda i: run_job(command, args.njobs, i, args.outputfile), range(args.njobs))
        )
    if any(returncodes):
        failed = [i for i, rc in enumerate(returncodes) if rc]
        print(f"{len(failed)} job(s) failed: {failed}; not merging, parts and logs kept")
        sys.exit(1)

    parts = [job_paths(args.outputfile, i)[0] for i in range(args.njobs)]
    print(f"all jobs done in {time.time() - start:.0f}s, merging into {args.outputfile}")
    merge = subprocess.run(["hadd", "-f", args.outputfile] + parts)
    if merge.returncode != 0:
        print("hadd failed; parts and logs kept")
        sys.exit(merge.returncode)
    if not args.keep_parts:
        for i in range(args.njobs):
            for path in job_paths(args.outputfile, i):
                os.remove(path)
    print(f"done in {time.time() - start:.0f}s")
