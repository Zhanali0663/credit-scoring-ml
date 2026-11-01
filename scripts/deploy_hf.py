"""Create and populate the Hugging Face Space for the credit scoring demo.

Usage::

    set HF_TOKEN=hf_xxx
    set HF_REPO_ID=<owner>/credit-scoring-demo
    .venv\\Scripts\\python -m scripts.deploy_hf

Pushes everything from ``deploy/hf_space/`` into a Docker Space. The build takes
~2-4 minutes. Free CPU Spaces may sleep or pause under quota; do not advertise
the URL as always-on.

Credentials: a user access token with **write** scope
(HF settings: https://huggingface.co/settings/tokens).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPACE_DIR = ROOT / "deploy" / "hf_space"
DEFAULT_REPO_ID = os.environ.get("HF_REPO_ID", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy the demo to a Hugging Face Docker Space")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN", ""),
        help="Hugging Face write token; falls back to the HF_TOKEN environment variable",
    )
    args = parser.parse_args()

    if not args.token:
        parser.error("No token supplied. Pass --token hf_xxx or set the HF_TOKEN environment variable")
    if not args.repo_id:
        parser.error("No space specified. Pass --repo-id <user>/<space> or set HF_REPO_ID")

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    who = api.whoami()
    print(f"Authenticated as: {who.get('name', who)}")

    repo_id = args.repo_id if "/" in args.repo_id else f"{who['name']}/{args.repo_id}"

    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=False,
        exist_ok=True,
    )
    print(f"Space ready: https://huggingface.co/spaces/{repo_id}")

    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(SPACE_DIR),
        commit_message="Deploy credit-scoring demo (LightGBM artifact + Docker app)",
    )
    print("Upload complete. The Space builds in ~2-4 min; the free tier may sleep or pause.")
    print(f"Live once built: https://{repo_id.replace('/', '-')}.hf.space")


if __name__ == "__main__":
    main()
