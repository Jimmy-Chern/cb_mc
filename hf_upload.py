#!/usr/bin/env python3
"""HF上传 — 带显式代理 + 断点续传 + 自动重试"""
import os, sys, time
os.environ['https_proxy'] = 'http://127.0.0.1:7899'
os.environ['http_proxy'] = 'http://127.0.0.1:7899'

from huggingface_hub import HfApi, create_repo, upload_file

TOKEN = '<HF_TOKEN>'
REPO = 'moonmonster/cb_mc_data'
FILE = '/tmp/cb_mc_data.tar.gz'

api = HfApi(token=TOKEN)
create_repo(repo_id=REPO, repo_type='dataset', private=True, exist_ok=True, token=TOKEN)

for attempt in range(1, 100):
    print(f"尝试 {attempt}: ", end='', flush=True)
    try:
        upload_file(
            path_or_fileobj=FILE,
            path_in_repo='cb_mc_data.tar.gz',
            repo_id=REPO,
            repo_type='dataset',
            token=TOKEN,
            run_as_future=False,
        )
        print("DONE!")
        sys.exit(0)
    except Exception as e:
        print(f"fail ({type(e).__name__}: {str(e)[:80]}), retry in 3s...")
        time.sleep(3)
