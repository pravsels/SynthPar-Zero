import json
from huggingface_hub import HfApi

# Load the access token from hf_token.json
with open('hf_token.json') as f:
    token_data = json.load(f)
    access_token = token_data['access_token']

repo_name = 'pravsels/synthpar'
dataset_folder = 'generated_images'

api = HfApi(token=access_token)

# Upload the entire folder to the Hugging Face Hub
api.upload_folder(
    folder_path=dataset_folder,
    path_in_repo='.',
    repo_id=repo_name,
    repo_type='dataset',
    token=access_token
)

print("Dataset uploaded successfully!")

