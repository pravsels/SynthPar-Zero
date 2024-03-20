import json
from huggingface_hub import HfApi

# Load the access token from hf_token.json
with open('hf_token.json') as f:
    token_data = json.load(f)
    access_token = token_data['access_token']

repo_name = 'pravsels/synthpar'
dataset_folder = 'generated_images'

api = HfApi(token=access_token)

api.upload_folder(
    folder_path=dataset_folder,
    repo_id=repo_name,
    repo_type="dataset",
    ignore_patterns=".gitignore",
    commit_message="Upload dataset"
)

print("Dataset uploaded successfully!")
