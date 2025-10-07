from huggingface_hub import snapshot_download
from pathlib import Path

datasets = [
    "lulululuyi/R-HORIZON-Math500",
    "lulululuyi/R-HORIZON-AIME24",
    "lulululuyi/R-HORIZON-AIME25",
    "lulululuyi/R-HORIZON-AMC23",
    "lulululuyi/R-HORIZON-Websearch",
]

for repo_id in datasets:
    dataset_name = repo_id.split("/")[-1]
    local_dir = Path("./evaluation/data") / dataset_name
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir)
    )