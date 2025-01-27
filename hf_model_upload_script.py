import json
from huggingface_hub import upload_file

# Load the access token from hf_token.json
with open('hf_token.json') as f:
    token_data = json.load(f)
    access_token = token_data['access_token']

repo_name = 'pravsels/synthpar'
model_file = 'models/network-conditional.pkl'

# Upload the local model file to the HF Hub
upload_file(
    path_or_fileobj=model_file,
    path_in_repo='network-conditional.pkl',
    repo_id=repo_name,
    repo_type='model',
    token=access_token
)
