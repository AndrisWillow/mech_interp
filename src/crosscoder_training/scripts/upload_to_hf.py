from huggingface_hub import HfApi, Repository
from pathlib import Path
import shutil

# TODO make relative path
script_dir = Path(__file__).parent.resolve()
SAVE_DIR = script_dir / "workspace" / "crosscoder-model-diff-replication" / "checkpoints"

model_name = "AndrisWillow/Qwen2.5-0.5B-crosscoder-13resid_pre" # to upload
local_model_dir = Path(f"{SAVE_DIR}/version_1")
files_to_upload = ["3.pt", "3_cfg.json"]

api = HfApi()
api.create_repo(repo_id=model_name, exist_ok=True)

hf_dir = Path("/tmp/hf_upload")
if hf_dir.exists():
    shutil.rmtree(hf_dir)
repo = Repository(local_dir=hf_dir, clone_from=model_name)

for fname in files_to_upload:
    src = local_model_dir / fname
    dst = hf_dir / fname
    shutil.copy(src, dst)

repo.push_to_hub(commit_message="Added Qwen Crosscoder")
