from huggingface_hub import snapshot_download

print("Downloading HeartMuLaGen...")
snapshot_download(
    repo_id="HeartMuLa/HeartMuLaGen",
    local_dir="./ckpt",
    local_dir_use_symlinks=False
)

print("Downloading HeartMuLa-oss-3B...")
snapshot_download(
    repo_id="HeartMuLa/HeartMuLa-oss-3B",
    local_dir="./ckpt/HeartMuLa-oss-3B",
    local_dir_use_symlinks=False
)

print("Downloading HeartCodec-oss...")
snapshot_download(
    repo_id="HeartMuLa/HeartCodec-oss",
    local_dir="./ckpt/HeartCodec-oss",
    local_dir_use_symlinks=False
)

print("All models downloaded successfully!")