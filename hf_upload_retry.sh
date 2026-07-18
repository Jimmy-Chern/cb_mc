#!/bin/bash
# HF上传自动重试 — 超时就kill重来, 利用HF分块续传
set -u
export https_proxy=http://127.0.0.1:7899
export http_proxy=http://127.0.0.1:7899
export ALL_PROXY=http://127.0.0.1:7899
export HF_TOKEN=<HF_TOKEN>
export HF_HUB_ENABLE_HF_TRANSFER=0

REPO="moonmonster/cb_mc_data"
FILE="/tmp/cb_mc_data.tar.gz"
HF=/home/xujiayang2/chenjunming/cb_mc/.venv/bin/hf
PER_TRY_TIMEOUT=90   # 每次尝试最多90秒, 超时kill重来
MAX_TRIES=50

for i in $(seq 1 $MAX_TRIES); do
    echo "===== 尝试 $i/$MAX_TRIES ($(date +%H:%M:%S)) ====="
    timeout $PER_TRY_TIMEOUT $HF upload "$REPO" "$FILE" cb_mc_data.tar.gz --repo-type dataset 2>&1 | grep -vE "conda|plugin|CONDA|package cache|envs|platform|user-agent|UID|netrc|offline|channel|repo.anaconda|^$"
    rc=${PIPESTATUS[0]}
    if [ $rc -eq 0 ]; then
        echo "===== 上传成功! ====="
        exit 0
    fi
    echo "  (超时/失败 rc=$rc, 续传重试...)"
    sleep 2
done
echo "===== 达到最大重试次数, 仍未成功 ====="
exit 1
