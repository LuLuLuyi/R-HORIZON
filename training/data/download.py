from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="lulululuyi/R-HORIZON-training-data",
    repo_type="dataset",
    local_dir="./training/data",
)