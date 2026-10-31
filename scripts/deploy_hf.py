"""Create and populate the Hugging Face Space for the credit scoring demo.

Usage::

    set HF_TOKEN=hf_xxx           # Windows
    .venv\\Scripts\\python -m scripts.deploy_hf

 pushes everything from ``deploy/hf_space/`` into a Streamlit Space named
 ``<owner>/credit-scoring-demo``. The Space build takes ~2-4 minutes; the app
 then stays online 24/7 on the free CPU tier.

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
DEFAULT_REPO_ID = "kuandykovzhanali/credit-scoring-demo"


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy the Streamlit demo to Hugging Face")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN", ""),
        help="Hugging Face write token; falls back to the HF_TOKEN environment variable",
    )
    args = parser.parse_args()

    if not args.token:
        parser.error("No token supplied. Pass --token hf_xxx or set the HF_TOKEN environment variable")

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    who = api.whoami()
    print(f"Authenticated as: {who.get('name', who)}")

    repo_id = args.repo_id if "/" in args.repo_id else f"{who['name']}/{args.repo_id}"

    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="gradio",
        private=False,
        exist_ok=True,
    )
    print(f"Space ready: https://huggingface.co/spaces/{repo_id}")

    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(SPACE_DIR),
        commit_message="Deploy credit-scoring demo (LightGBM artifact + Streamlit app)",
    )
    print("Upload complete. The Space builds in ~2-4 min, then stays online 24/7.")
    print(f"Live once built: https://{repo_id.replace('/', '-')}.hf.space")


if __name__ == "__main__":
    main()
