import os 
from huggingface_hub import CommitOperationAdd, preupload_lfs_files, create_commit 
import time 
from zipfile import ZipFile 
from tqdm import tqdm 
import json 

with open('hf_token.json') as f:
    token_data = json.load(f)
    access_token = token_data['access_token']

images_per_shard = 10000
dataset_folder = "generated_images"
repo_id = 'pravsels/synthpar'

image_files = [f for f in os.listdir(dataset_folder) if f.endswith((".jpg", ".png", ".jpeg"))]
total_shards = (len(image_files) + images_per_shard - 1) // images_per_shard

operations = []  
shard_file_names = []
shard_folder = 'dataset_shards'
os.makedirs(shard_folder, exist_ok=True)
for shard_index in tqdm(range(total_shards), desc='creating shards...'):
    start_index = shard_index * images_per_shard
    end_index = min((shard_index + 1) * images_per_shard, len(image_files))
    shard_files = image_files[start_index:end_index]
    
    shard_file_name = f"{shard_folder}/shard_{shard_index+1}_of_{total_shards}.zip"
    shard_file_names.append(shard_file_name)
    
    if not os.path.exists(shard_file_name):
        with ZipFile(shard_file_name, "w") as zip_file:
            for image_file in shard_files:
                image_path = os.path.join(dataset_folder, image_file)
                zip_file.write(image_path, image_file)
    

for shard_file_name in tqdm(shard_file_names, desc='uploading shards...'):
    max_retries = 10
    retry_delay = 60
    for retry_attempt in range(max_retries):
        try:
            # Create a CommitOperationAdd object for the shard
            addition = CommitOperationAdd(path_in_repo=shard_file_name.split('/')[-1], 
                                          path_or_fileobj=shard_file_name)
            preupload_lfs_files(repo_id=repo_id,
                                repo_type='dataset', 
                                token=access_token,
                                additions=[addition])
            operations.append(addition)
            break  # Exit the retry loop if the upload is successful
        except Exception as e:
            print(f"Error uploading shard {shard_index}: {str(e)}")
            if retry_attempt < max_retries - 1:
                print(f"Retrying upload in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Max retries exceeded. Skipping shard {shard_index}.")

try:
    create_commit(repo_id=repo_id, 
                  repo_type='dataset',
                  token=access_token,
                  operations=operations, 
                  commit_message="Upload image dataset")
    print("Image dataset uploaded successfully.")
except Exception as e:
    print(f"Error creating commit: {str(e)}")

for shard_file_name in tqdm(shard_file_names, desc='deleting shards...'):
    os.remove(shard_file_name)
os.rmdir(shard_folder)
