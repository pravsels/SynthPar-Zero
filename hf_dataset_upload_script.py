import os
import json
from huggingface_hub import HfApi
from tqdm import tqdm 

# Load the access token from hf_token.json
with open('hf_token.json') as f:
    token_data = json.load(f)
    access_token = token_data['access_token']

repo_name = 'pravsels/synpar'
dataset_folder = 'generated_images'
# (number of files to upload per batch)
batch_size = 128

api = HfApi(token=access_token)

# Get the list of all files in the dataset folder and its subfolders
file_list = []
for root, dirs, files in os.walk(dataset_folder):
    for file_name in files:
        file_path = os.path.join(root, file_name)
        file_list.append(file_path)

# Calculate the total number of batches
total_batches = (len(file_list) + batch_size - 1) // batch_size

for i in tqdm(range(total_batches)):
    start_index = i * batch_size
    end_index = min((i + 1) * batch_size, len(file_list))
    batch_files = file_list[start_index:end_index]
    
    for file_path in batch_files:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_path,
            repo_id=repo_name,
            repo_type='dataset',
            token=access_token
        )

print("Dataset uploaded successfully!")

