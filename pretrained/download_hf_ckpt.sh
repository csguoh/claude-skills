#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Error: At least one argument is required."
    echo "Usage: $0 <repo_id> [repo_type] [local_dir]"
    exit 1
fi

export HF_ENDPOINT=https://hf-mirror.com

repo_id=$1
repo_type=${2:-"model"}
local_dir=${3:-"./${1##*/}"}

# 关键修复：创建目录
mkdir -p "$local_dir"
echo "Directory ensured: $local_dir"

echo "Starting download:"
echo "  repo_id:   $repo_id"
echo "  repo_type: $repo_type"
echo "  local_dir: $local_dir"

python3 download_hf_ckpt.py "$repo_id" "$repo_type" "$local_dir"
