import os
from huggingface_hub import hf_hub_download
from zipfile import ZipFile
from tqdm import tqdm

hf_repo_id = "pravsels/synthpar"
dataset_path = "../../datasets"
dataset_dir = os.path.join(dataset_path, "synthpar")
os.makedirs(dataset_dir, exist_ok=True)

num_shards = 8
for shard_num in range(1, num_shards + 1):
    shard_filename = f"shard_{shard_num}_of_{num_shards}.zip"
    shard_path = hf_hub_download(repo_id=hf_repo_id,
                                 repo_type="dataset",
                                 filename=shard_filename,
                                 local_dir=dataset_path)
    
    with ZipFile(shard_path, "r") as zip_ref:
        total_files = len(zip_ref.namelist())
        progress_bar = tqdm(total=total_files,
                            unit="file",
                            desc=f"Extracting images from shard {shard_num}")
        
        sorted_filenames = sorted(zip_ref.namelist())
        for file in sorted_filenames:
            zip_ref.extract(file, dataset_dir)
            progress_bar.update(1)
        
        progress_bar.close()
    
    os.remove(shard_path)

