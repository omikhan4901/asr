#!/usr/bin/env python3
"""
Copy the finished OpenSLR 53 training zip and its lists from the friend's Modal volume to another account's
volume, through this PC (2026-10-20: openslr.org was too slow to rebuild it on the new account, ~0.5 MB/s).
The zip lands in data/backup/zips/ (where runs/backup_data.py expects it, so the backup skips it later).

  ./Scripts/python.exe runs/copy_openslr_volume.py --plan     # sizes and what would be copied (read-only)
  ./Scripts/python.exe runs/copy_openslr_volume.py            # download, then upload; run again to resume

A finished step is skipped on a re-run (sizes are compared); an interrupted download starts that file again.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOLUME = "bangla-asr-ft"
FILES = ["zips/openslr53-train.zip", "manifests/matrix/openslr53-train.csv",
         "manifests/matrix/openslr53-train_report.json", "manifests/matrix/m5-300.csv"]


def local_path(f: str, src: str) -> Path:
    if f.startswith("zips/"):
        return ROOT / "data" / "backup" / f
    return ROOT / "data" / "backup" / f"modal_{src}" / f


# ---- child steps: each runs in its own process so MODAL_PROFILE picks the account
def child(step: str, src: str):
    import modal
    vol = modal.Volume.from_name(VOLUME)
    if step == "sizes":
        out = {}
        for f in FILES:
            d = str(Path(f).parent.as_posix())
            out.update({e.path.strip("/"): e.size for e in vol.listdir(d) if e.path.strip("/") in FILES})
        print(json.dumps(out))
    elif step == "down":
        for f in FILES:
            dst = local_path(f, src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            t0, n, mark = time.time(), 0, 0
            with tmp.open("wb") as fh:
                for chunk in vol.read_file(f):
                    fh.write(chunk)
                    n += len(chunk)
                    if n - mark >= 250e6:
                        mark = n
                        print(f"  {f}: {n / 1e9:.2f} GB, {n / 1e6 / (time.time() - t0):.1f} MB/s", flush=True)
            tmp.replace(dst)
            print(f"  downloaded {f} ({n / 1e9:.2f} GB, {time.time() - t0:.0f} s)", flush=True)
    elif step == "up":
        t0 = time.time()
        with vol.batch_upload(force=True) as batch:
            for f in FILES:
                batch.put_file(str(local_path(f, src)), "/" + f)
        print(f"  uploaded {len(FILES)} files ({time.time() - t0:.0f} s)", flush=True)


def run(step: str, profile: str, src: str, files=None) -> str:
    env = {**os.environ, "MODAL_PROFILE": profile, "PYTHONIOENCODING": "utf-8"}
    cmd = [sys.executable, "-W", "ignore", __file__, "--child", step, "--src", src]
    if step == "sizes":
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8")
        if r.returncode:
            sys.exit(f"listing the {profile} volume failed:\n{r.stderr[-1500:]}")
        return r.stdout.strip().splitlines()[-1]
    if subprocess.run(cmd, env=env).returncode:
        sys.exit(f"step '{step}' on {profile} failed: run the same command again to continue")
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="friend")
    ap.add_argument("--dst", default="newacct")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--child", default="")
    a = ap.parse_args()
    if a.child:
        return child(a.child, a.src)

    src = json.loads(run("sizes", a.src, a.src))
    missing = [f for f in FILES if f not in src]
    if missing:
        sys.exit(f"not on the {a.src} volume: {missing}")
    dst = json.loads(run("sizes", a.dst, a.src))
    have_local = all(local_path(f, a.src).exists() and local_path(f, a.src).stat().st_size == src[f] for f in FILES)
    on_dst = all(dst.get(f) == src[f] for f in FILES)
    print(f"{'file':48s} {a.src:>10s} {'this PC':>10s} {a.dst:>10s}")
    for f in FILES:
        p = local_path(f, a.src)
        print(f"{f:48s} {src[f] / 1e6:9.1f}M {(p.stat().st_size / 1e6 if p.exists() else 0):9.1f}M "
              f"{(dst.get(f) or 0) / 1e6:9.1f}M")
    if on_dst:
        print(f"\nalready on the {a.dst} volume: nothing to do.")
        return
    if a.plan:
        print(f"\nplan: {'skip download (already on this PC)' if have_local else 'download'} -> upload to {a.dst}")
        return
    if not have_local:
        print(f"\n== download from {a.src} (~{src[FILES[0]] / 1e9:.1f} GB)", flush=True)
        run("down", a.src, a.src)
    bad = [f for f in FILES if local_path(f, a.src).stat().st_size != src[f]]
    if bad:
        sys.exit(f"size mismatch after download: {bad}; run again")
    print(f"\n== upload to {a.dst} (no progress bar; same size as the download, keep the window open)", flush=True)
    run("up", a.dst, a.src)
    dst = json.loads(run("sizes", a.dst, a.src))
    if any(dst.get(f) != src[f] for f in FILES):
        sys.exit(f"after upload the sizes differ on {a.dst}: {dst}; run again")
    print(f"\ndone: all {len(FILES)} files on the {a.dst} volume, same sizes as on {a.src}. Now start M5-300:\n"
          "  ./Scripts/python.exe pipeline/modal_launch.py train --method full --run m5_300 --source m5_300 --budget 15.5")


if __name__ == "__main__":
    main()
