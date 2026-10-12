#!/usr/bin/env bash
# Send the code of the current commit (src/ + pipeline/) to the private hub, so the Kaggle notebook runs it.
# Same result as `modal run pipeline/modal_finetune.py::to_hf --what code`, but from this PC and free.
# Needs: committed code, and `./Scripts/hf.exe auth login` done with the write token.
#   bash runs/upload_code_to_hub.sh
set -eu
cd "$(dirname "$0")/.."
if [ -n "$(git status --porcelain src pipeline)" ]; then
  echo "Commit the code first: Kaggle gets the code of the current commit." >&2
  exit 1
fi
C=$(git rev-parse HEAD | cut -c1-12)
TMP=$(mktemp -d)
git archive --format=tar.gz HEAD src pipeline > "$TMP/$C.tar.gz"
printf '%s' "$C" > "$TMP/latest.txt"
PYTHONIOENCODING=utf-8 C="$C" TMP="$TMP" ./Scripts/python.exe - <<'EOF'
import os
from huggingface_hub import HfApi, hf_hub_download
api, c, t, R = HfApi(), os.environ["C"], os.environ["TMP"], "OmiKhan/bangla-asr-hub"
api.upload_file(repo_id=R, repo_type="dataset", path_or_fileobj=os.path.join(t, c + ".tar.gz"), path_in_repo=f"code/{c}.tar.gz",
                commit_message=f"code {c}")
api.upload_file(repo_id=R, repo_type="dataset", path_or_fileobj=os.path.join(t, "latest.txt"), path_in_repo="code/latest.txt",
                commit_message=f"code/latest.txt = {c}")
got = open(hf_hub_download(R, "code/latest.txt", repo_type="dataset", force_download=True)).read().strip()
print(f"hub code/latest.txt = {got}", "(OK)" if got == c else "(MISMATCH)")
EOF
rm -rf "$TMP"
