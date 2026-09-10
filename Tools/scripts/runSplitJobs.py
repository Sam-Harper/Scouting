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
Progress is shown live by tailing the logs for "Processed X/Y events"
lines (per-job block on a terminal, periodic summary lines otherwise).
Parts and logs are deleted after a successful merge unless --keep-parts.
"""
import argparse
import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
import time

PROGRESS_RE = re.compile(r"Processed (\d+)/(\d+) events")


def job_paths(outputfile, jobnr):
    base, ext = os.path.splitext(outputfile)
    return f"{base}_job{jobnr}{ext}", f"{base}_job{jobnr}.log"


def last_progress(logfile):
    """Return (events done, events total) from the log's last progress line."""
    try:
        with open(logfile, "rb") as log:
            log.seek(0, os.SEEK_END)
            log.seek(max(0, log.tell() - 4096))
            tail = log.read().decode(errors="replace")
    except OSError:
        return None
    matches = PROGRESS_RE.findall(tail)
    return (int(matches[-1][0]), int(matches[-1][1])) if matches else None


def run_job(command, njobs, jobnr, outputfile, status):
    status["start"] = time.time()
    status["state"] = "running"
    partfile, logfile = job_paths(outputfile, jobnr)
    cmd = command + ["--njobs", str(njobs), "--jobnr", str(jobnr), "-o", partfile]
    with open(logfile, "w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    status["time"] = time.time() - status["start"]
    status["rc"] = result.returncode
    status["state"] = "done" if result.returncode == 0 else "failed"
    return result.returncode


def status_lines(statuses, outputfile, start):
    """One status line per job plus an aggregate total line."""
    lines = []
    events_done = events_total = 0
    for jobnr, status in enumerate(statuses):
        logfile = job_paths(outputfile, jobnr)[1]
        progress = None if status["state"] == "queued" else last_progress(logfile)
        if progress:
            events_done += progress[0]
            events_total += progress[1]
        if status["state"] == "queued":
            lines.append(f"job {jobnr}: queued")
        elif status["state"] == "running":
            if progress:
                lines.append(f"job {jobnr}: {progress[0]}/{progress[1]} events")
            else:
                lines.append(f"job {jobnr}: starting")
        elif status["state"] == "done":
            lines.append(f"job {jobnr}: done in {status['time']:.0f}s")
        else:
            lines.append(f"job {jobnr}: FAILED (exit {status['rc']}, see {logfile})")
    njobs_ended = sum(status["state"] in ("done", "failed") for status in statuses)
    elapsed = time.time() - start
    lines.append(
        f"total: {njobs_ended}/{len(statuses)} jobs done, "
        f"{events_done}/{events_total} events ({events_done / elapsed:.0f} ev/s, {elapsed:.0f}s elapsed)"
    )
    return lines


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
    statuses = [{"state": "queued", "rc": None, "time": None} for _ in range(args.njobs)]
    start = time.time()
    is_tty = sys.stdout.isatty()
    poll_interval = 2.0 if is_tty else 15.0
    with concurrent.futures.ThreadPoolExecutor(max_parallel) as pool:
        futures = [
            pool.submit(run_job, command, args.njobs, jobnr, args.outputfile, statuses[jobnr])
            for jobnr in range(args.njobs)
        ]
        drawn_lines = 0
        reported = set()
        while True:
            all_ended = all(future.done() for future in futures)
            lines = status_lines(statuses, args.outputfile, start)
            if is_tty:
                # redraw the block in place: move up over the previous one
                if drawn_lines:
                    sys.stdout.write(f"\x1b[{drawn_lines}A")
                sys.stdout.write("".join(f"\x1b[2K{line}\n" for line in lines))
                sys.stdout.flush()
                drawn_lines = len(lines)
            else:
                for jobnr, status in enumerate(statuses):
                    if status["state"] in ("done", "failed") and jobnr not in reported:
                        reported.add(jobnr)
                        print(lines[jobnr], flush=True)
                print(lines[-1], flush=True)
            if all_ended:
                break
            concurrent.futures.wait(futures, timeout=poll_interval)
    returncodes = [future.result() for future in futures]
    if any(returncodes):
        failed = [jobnr for jobnr, rc in enumerate(returncodes) if rc]
        print(f"{len(failed)} job(s) failed: {failed}; not merging, parts and logs kept")
        sys.exit(1)

    parts = [job_paths(args.outputfile, jobnr)[0] for jobnr in range(args.njobs)]
    print(f"merging into {args.outputfile}")
    merge = subprocess.run(["hadd", "-f", args.outputfile] + parts)
    if merge.returncode != 0:
        print("hadd failed; parts and logs kept")
        sys.exit(merge.returncode)
    if not args.keep_parts:
        for jobnr in range(args.njobs):
            for path in job_paths(args.outputfile, jobnr):
                os.remove(path)
    print(f"done in {time.time() - start:.0f}s")
