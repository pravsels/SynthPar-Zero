import os
import shutil
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms
from PIL import Image
from utils import class_to_race_map
from huggingface_hub import hf_hub_download
from tqdm import tqdm

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, category, transform=None):
        self.root_dir = root_dir
        self.category = category
        self.transform = transform
        self.image_paths = self.get_image_paths()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image_path

    def get_image_paths(self):
        category_dir = os.path.join(self.root_dir, self.category)
        image_files = [f for f in os.listdir(category_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_paths = [os.path.join(category_dir, f) for f in image_files]
        return image_paths

class FairFacePredictor:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.transform = self.get_transform()
        self.race_labels = ['White', 'Black', 'Asian', 'Indian']
        self.gender_labels = ['Male', 'Female']

    def load_model(self, repo_id='pravsels/synpar', 
                         model_dir='models',
                         model_filename='fairface_race4.pt'):
        
        print(f'Downloading model from Hugging Face Hub... ', end='', flush=True)
        local_model_file = hf_hub_download(repo_id=repo_id, 
                                            local_dir=model_dir,
                                            local_dir_use_symlinks="auto",
                                            filename=model_filename)

        model = torchvision.models.resnet34(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 18)
        model.load_state_dict(torch.load(local_model_file))
        model = model.to(self.device)
        model.eval()
        return model

    def get_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])


if __name__ == "__main__":
    root_dir = './generated_images'
    filtered_dir = './filtered_images'
    imgs_to_filter = 10000   # stopping criteria (positive filters)
    batch_size = 256

    predictor = FairFacePredictor()

    for class_label, class_name in class_to_race_map.items():
        print(f'filtering images for {class_name}...')

        no_of_imgs_accepted = 0
        no_of_imgs_rejected = 0 

        dataset = ImageDataset(root_dir, 
                               class_name, 
                               transform=predictor.transform)
        
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=batch_size, 
                                                 shuffle=False)

        filtered_class_dir = os.path.join(filtered_dir, class_name)
        os.makedirs(filtered_class_dir, exist_ok=True)

        accepted_dir = os.path.join(filtered_class_dir, 'accepted')
        rejected_dir = os.path.join(filtered_class_dir, 'rejected')
        os.makedirs(accepted_dir, exist_ok=True)
        os.makedirs(rejected_dir, exist_ok=True)

        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f'processed {no_of_imgs_accepted + no_of_imgs_rejected} images for {class_name}')
            for batch_images, image_paths in pbar:
                batch_images = batch_images.to(predictor.device)
                outputs = predictor.model(batch_images)
                outputs = outputs.cpu().numpy()

                race_outputs = outputs[:, :4]
                race_scores = np.exp(race_outputs) / np.sum(np.exp(race_outputs), 
                                                            axis=1, 
                                                            keepdims=True)
                race_preds = [predictor.race_labels[np.argmax(scores)] for scores in race_scores]

                gender_outputs = outputs[:, 4:6]
                gender_scores = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs), 
                                                                axis=1, 
                                                                keepdims=True)
                gender_preds = [predictor.gender_labels[np.argmax(scores)] for scores in gender_scores]

                for (race_pred, gender_pred), image_path in zip(zip(race_preds, gender_preds), image_paths):
                    
                    folder_category = os.path.basename(os.path.dirname(image_path))

                    if race_pred + gender_pred == folder_category:
                        accepted_path = os.path.join(accepted_dir, f"{os.path.basename(image_path)}_{race_pred}_{gender_pred}.png")
                        shutil.copy(image_path, accepted_path)
                        # keep count of no of positively filtered images 
                        no_of_imgs_accepted += 1 
                    else:
                        rejected_path = os.path.join(rejected_dir, f"{os.path.basename(image_path)}_{race_pred}_{gender_pred}.png")
                        shutil.copy(image_path, rejected_path)

                        no_of_imgs_rejected += 1

                    pbar.set_description(f'processed {no_of_imgs_accepted + no_of_imgs_rejected} images for {class_name}')

            print('No of images filtered through : ', no_of_imgs_accepted)
            print('No of images rejected : ', no_of_imgs_rejected)

