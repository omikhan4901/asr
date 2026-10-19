#!/usr/bin/env python3
"""
Start Phase 4 jobs on the DEPLOYED Modal app, so they keep running when every window on this PC closes.

`modal run --detach` keeps a client on this PC waiting for the job; killing that client (closing
VS Code or the terminal) cancelled every job on 2026-10-10. Here the app is deployed once, and each
job is spawned on it: Modal runs it to the end with nothing waiting on this PC. Each launch is
appended to logs/modal_calls.jsonl (call id, arguments, git state).

  ./Scripts/python.exe pipeline/modal_launch.py deploy                  (after any code change)
  ./Scripts/python.exe pipeline/modal_launch.py train --method decoder --run m1_decoder --budget 5
  ./Scripts/python.exe pipeline/modal_launch.py train --method full --run m2_cv --source cv --budget 3
  ./Scripts/python.exe pipeline/modal_launch.py score --run m1_full --model checkpoints/m1_full/final --budget 3
  ./Scripts/python.exe pipeline/modal_launch.py score --run m1_lora --adapter checkpoints/m1_lora/final --budget 3
  ./Scripts/python.exe pipeline/modal_launch.py score --run tugstugi_a100 --out zeroshot --budget 3
  ./Scripts/python.exe pipeline/modal_launch.py prep --sources kathbath,shrutilipi   (matrix data, CPU only)
  ./Scripts/python.exe pipeline/modal_launch.py status                  (state of every launched job)
Live logs: Modal website -> Apps -> bangla-asr-finetune (deployed) -> the function's calls.
Uploads and downloads stay in modal_finetune.py (modal run ...::upload / pull / pull_results).
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modal  # noqa: E402

from modal_finetune import APP, SHIPPED, TEST_SETS, TRAIN_SOURCES, TUGSTUGI, _git_state  # noqa: E402

CALLS = Path("logs/modal_calls.jsonl")
DEPLOYED = Path("logs/modal_deployed.json")


def _profile() -> str:
    """The Modal account (profile) commands go to; deployments are per account."""
    out = subprocess.run([str(Path("Scripts/modal.exe")), "profile", "current"], capture_output=True, text=True)
    return out.stdout.strip()


def deploy():
    state = {**_git_state(), "profile": _profile()}
    if state["git_dirty_shipped"]:
        sys.exit("Commit the changes to the shipped files first (src/bangla_asr, pipeline/*.py used on Modal), "
                 "so every job can be traced to a commit.")
    code = subprocess.run([str(Path("Scripts/modal.exe")), "deploy", "pipeline/modal_finetune.py"]).returncode
    if code:
        sys.exit(f"modal deploy failed (exit {code})")
    DEPLOYED.parent.mkdir(parents=True, exist_ok=True)
    deployed = json.loads(DEPLOYED.read_text()) if DEPLOYED.exists() else {}
    deployed = deployed if "by_profile" in deployed else {"by_profile": {}}
    deployed["by_profile"][state["profile"]] = {**state, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    DEPLOYED.write_text(json.dumps(deployed, indent=2))
    print(f"deployed commit {state['git_commit'][:7]} to Modal account '{state['profile']}'")


def _check_deployed() -> dict:
    """The deployed code must match HEAD in the shipped files, otherwise the launch record would lie.
    Commits that touch only other files (docs, analysis) do not need a new deploy."""
    profile = _profile()
    dep = json.loads(DEPLOYED.read_text()).get("by_profile", {}).get(profile) if DEPLOYED.exists() else None
    if dep is None:
        sys.exit(f"Not deployed to Modal account '{profile}' yet: run  pipeline/modal_launch.py deploy")
    now = _git_state()
    same = now["git_commit"] == dep["git_commit"] or subprocess.run(
        ["git", "diff", "--quiet", dep["git_commit"], "HEAD", "--", *SHIPPED]).returncode == 0
    if now["git_dirty_shipped"] or not same:
        sys.exit(f"The deployed code ({dep['git_commit'][:7]}) is not the current code ({now['git_commit'][:7]}, "
                 f"uncommitted changes: {now['git_dirty_shipped']}). Commit, then run  modal_launch.py deploy.")
    return {**now, "profile": profile}


def _spawn(fn_name: str, kind: str, record: dict, *args, **kwargs):
    launch = _check_deployed()
    call = modal.Function.from_name(APP, fn_name).spawn(*args, launch, **kwargs)
    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "call_id": call.object_id, "kind": kind, **record,
             "git_commit": launch["git_commit"], "profile": launch["profile"]}
    CALLS.parent.mkdir(parents=True, exist_ok=True)
    with CALLS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"started {kind} {record['run']} -> call {call.object_id}. Nothing needs to stay open on this PC.")


def status():
    if not CALLS.exists():
        sys.exit("no jobs launched yet")
    profile, other = _profile(), 0
    for line in CALLS.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e.get("profile") != profile:  # calls of another account cannot be looked up from this one
            other += 1
            continue
        try:
            out = modal.FunctionCall.from_id(e["call_id"]).get(timeout=0)
            state = f"finished, exit code {out}"
        except TimeoutError:
            state = "running"
        except Exception as ex:  # cancelled, failed, or results expired (7 days)
            state = f"{type(ex).__name__}: {str(ex)[:80]}"
        print(f"{e['time']}  {e['kind']:5}  {e['run']:16} {state}")
    print(f"(account '{profile}'; {other} older jobs of other accounts not shown)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("deploy")
    sub.add_parser("status")
    t = sub.add_parser("train")
    t.add_argument("--method", required=True, choices=["lora", "decoder", "encoder", "full"])
    t.add_argument("--run", required=True)
    t.add_argument("--budget", type=float, required=True, help="dollar cap for the whole run (earlier calls count)")
    t.add_argument("--extra", default="", help="more 11_finetune.py arguments, space-separated")
    t.add_argument("--source", default="dlsprint", choices=sorted(TRAIN_SOURCES),
                   help="training data (matrix runs); the audio must be on this account's volume (from_hf, prep_matrix)")
    pr = sub.add_parser("prep")
    pr.add_argument("--sources", default="kathbath,shrutilipi")
    pr.add_argument("--max_hours", type=float, default=34.0, help="hours per source; 34 = all of CV train, so every matrix source is the same size")
    s = sub.add_parser("score")
    s.add_argument("--run", required=True)
    s.add_argument("--budget", type=float, required=True)
    s.add_argument("--model", default=TUGSTUGI, help="HF id or a volume path such as checkpoints/m1_full/final")
    s.add_argument("--adapter", default="", help="LoRA final/ on the volume")
    s.add_argument("--sets", default="all", help="comma list or 'all': " + ",".join(TEST_SETS))
    s.add_argument("--out", default="finetuned", help="results/<out>/ on the volume (zeroshot for the baseline)")
    a = p.parse_args()

    if a.cmd == "deploy":
        deploy()
    elif a.cmd == "status":
        status()
    elif a.cmd == "prep":
        _spawn("prep_matrix_remote", "prep", {"run": "matrix_data", "sources": a.sources, "max_hours": a.max_hours},
               a.sources.split(","), a.max_hours, True)
    elif a.cmd == "train":
        _spawn("train_remote", "train", {"run": a.run, "method": a.method, "budget": a.budget, "extra": a.extra,
                                         "source": a.source},
               a.method, a.run, a.extra.split(), a.budget, source=a.source)
    else:
        sets = TEST_SETS if a.sets == "all" else a.sets.split(",")
        if set(sets) - set(TEST_SETS):
            sys.exit(f"unknown sets {set(sets) - set(TEST_SETS)}; choose from {TEST_SETS}")
        _spawn("score_remote", "score", {"run": a.run, "model": a.model, "adapter": a.adapter, "sets": sets,
                                         "out": a.out, "budget": a.budget},
               a.run, a.model, a.adapter, sets, a.out, a.budget)


if __name__ == "__main__":
    main()
