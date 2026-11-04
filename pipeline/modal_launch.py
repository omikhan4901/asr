#!/usr/bin/env python3
"""
Start fine-tuning and scoring jobs on the DEPLOYED Modal app, so they keep running when every window on this PC closes.

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
  ./Scripts/python.exe pipeline/modal_launch.py prep-extra --source kathbath --max_hours 82 --set_name kathbath-full
  ./Scripts/python.exe pipeline/modal_launch.py numbers-align --limit 300            (number fix: readings vs audio)
  ./Scripts/python.exe pipeline/modal_launch.py numbers-judge                        (number fix: the model writes numbers in words)
  ./Scripts/python.exe pipeline/modal_launch.py numbers-mix                          (number fix: training list)
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
# The Modal CLI next to the Python running this script (.venv\Scripts on Windows, .venv/bin on Linux).
MODAL = str(Path(sys.executable).parent / ("modal.exe" if sys.platform == "win32" else "modal"))


def _profile() -> str:
    """The Modal account (profile) commands go to; deployments are per account."""
    out = subprocess.run([MODAL, "profile", "current"], capture_output=True, text=True)
    return out.stdout.strip()


def deploy():
    state = {**_git_state(), "profile": _profile()}
    if state["git_dirty_shipped"]:
        sys.exit("Commit the changes to the shipped files first (src/bangla_asr, pipeline/*.py used on Modal), "
                 "so every job can be traced to a commit.")
    code = subprocess.run([MODAL, "deploy", "pipeline/modal_finetune.py"]).returncode
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
    px = sub.add_parser("prep-extra", help="Kathbath or Shrutilipi under a new set name (does not touch the train pulls or m5 lists)")
    px.add_argument("--source", required=True, choices=["kathbath", "shrutilipi", "indicvoices"])
    px.add_argument("--max_hours", type=float, required=True)
    px.add_argument("--set_name", required=True)
    px.add_argument("--max_files", type=int, default=0, help="probe: only the first N parquet files")
    px.add_argument("--limit", type=int, default=0, help="probe: stop after N usable rows")
    na = sub.add_parser("numbers-align", help="score the spoken readings of every digit token against the audio")
    na.add_argument("--limit", type=int, default=0, help="random clips only (0 = all ~6,400 clips with digits)")
    sub.add_parser("numbers-judge", help="the released model writes every number clip in words (digits blocked)")
    nm = sub.add_parser("numbers-mix", help="training list: readings the judge wrote (default) or by MMS score")
    nm.add_argument("--min_score", type=float, default=0.4, help="MMS rule, only with --no_judge")
    nm.add_argument("--no_judge", action="store_true")
    nm.add_argument("--balance", action="store_true", help="filler keeps m5-300's source shares -> numbers-v2 (v1.2)")
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
    elif a.cmd == "prep-extra":
        _spawn("prep_extra_remote", "prep", {"run": f"prep_{a.set_name}", "source": a.source, "max_hours": a.max_hours, "set_name": a.set_name, "max_files": a.max_files, "limit": a.limit},
               a.source, a.max_hours, a.set_name, a.max_files, a.limit)
    elif a.cmd == "numbers-align":
        _spawn("numbers_align_remote", "prep", {"run": f"numbers_align_{a.limit or 'all'}", "limit": a.limit}, a.limit)
    elif a.cmd == "numbers-judge":
        _spawn("numbers_judge_remote", "prep", {"run": "numbers_judge"})
    elif a.cmd == "numbers-mix":
        _spawn("numbers_mix_remote", "prep", {"run": "numbers_mix" + ("_v2" if a.balance else ""), "judge": not a.no_judge,
                                              "min_score": a.min_score, "balance": a.balance},
               a.min_score, not a.no_judge, a.balance)
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
