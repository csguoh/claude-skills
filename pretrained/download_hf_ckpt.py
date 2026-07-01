import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 必须第一行！

from huggingface_hub import snapshot_download  # 之后再import
import sys

if __name__ == "__main__":
    
    repo_id = sys.argv[1]
    repo_type = sys.argv[2] if len(sys.argv) > 2 else "model"
    local_dir = sys.argv[3] if len(sys.argv) > 3 else f"./{repo_id.split('/')[-1]}"
    
    os.makedirs(local_dir, exist_ok=True)
    print(f"Directory ensured: {local_dir}")
    print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT')}")  
    print(f"Downloading {repo_type}: {repo_id} -> {local_dir}")
    
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=local_dir,
        endpoint="https://hf-mirror.com",  
    )
