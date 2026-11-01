"""Upload ``deploy/hf_space`` to a Hugging Face Space.

The token is never stored in the repository. Provide it through the
``HF_TOKEN`` environment variable (write scope) and the target through
``HF_REPO_ID`` (for example ``your-user/credit-scoring-demo``).

Usage::

    set HF_TOKEN=hf_xxx
    set HF_REPO_ID=your-user/credit-scoring-demo
    .venv\\Scripts\\python -m scripts.upload_new_space
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPACE_DIR = ROOT / "deploy" / "hf_space"


def main() -> None:
    token = os.environ.get("HF_TOKEN", "")
    repo_id = os.environ.get("HF_REPO_ID", "")
    if not token:
        raise SystemExit("HF_TOKEN is not set. Create a write token at https://huggingface.co/settings/tokens")
    if not repo_id:
        raise SystemExit("HF_REPO_ID is not set (expected '<user>/<space-name>').")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    who = api.whoami()
    print(f"Authenticated as: {who.get('name', who)}")
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(SPACE_DIR),
        commit_message="Deploy credit-scoring demo (LightGBM artifact + Docker app)",
    )
    print(f"Uploaded to https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
