import json
from huggingface_hub import upload_file

# Load the access token from hf_token.json
with open('hf_token.json') as f:
    token_data = json.load(f)
    access_token = token_data['access_token']

# Set the repository name and path to the local .pkl file
repo_name = 'pravsels/stylegan2_conditional'
pkl_file = 'models/network-conditional.pkl'

# Upload the local .pkl file to the Hugging Face Hub
upload_file(
    path_or_fileobj=pkl_file,
    path_in_repo='network-conditional.pkl',
    repo_id=repo_name,
    repo_type='model',
    token=access_token
)

