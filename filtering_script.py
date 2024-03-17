import os
import shutil
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import transforms
from PIL import Image

class FairFaceDataset(torch.utils.data.Dataset):
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
    def __init__(self, model_path, batch_size=32):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)
        self.transform = self.get_transform()
        self.race_labels = ['White', 'Black', 'Asian', 'Indian']
        self.gender_labels = ['Male', 'Female']
        self.batch_size = batch_size

    def load_model(self, model_path):
        model = torchvision.models.resnet34(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 18)
        model.load_state_dict(torch.load(model_path))
        model = model.to(self.device)
        model.eval()
        return model

    def get_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, dataloader):
        rejected_dir = 'rejected'
        os.makedirs(rejected_dir, exist_ok=True)

        with torch.no_grad():
            for batch_images, image_paths in dataloader:
                batch_images = batch_images.to(self.device)
                outputs = self.model(batch_images)
                outputs = outputs.cpu().numpy()

                for output, image_path in zip(outputs, image_paths):
                    race_outputs = output[:4]
                    gender_outputs = output[4:6]
                    race_pred = self.race_labels[np.argmax(race_outputs)]
                    gender_pred = self.gender_labels[np.argmax(gender_outputs)]

                    folder_category = os.path.basename(os.path.dirname(image_path))
                    expected_race, expected_gender = folder_category.split('_')

                    if race_pred != expected_race or gender_pred != expected_gender:
                        rejected_path = os.path.join(rejected_dir, f"{os.path.basename(image_path)}_{race_pred}_{gender_pred}.png")
                        shutil.copy(image_path, rejected_path)


if __name__ == "__main__":
    model_path = './models/fairface_race4.pt'
    batch_size = 4
    root_dir = './generated_images'

    categories = ['IndianFemale', 'BlackFemale', 'AsianFemale', 'AsianMale',
                  'WhiteMale', 'IndianMale', 'BlackMale', 'WhiteFemale']

    predictor = FairFacePredictor(model_path, 
                                  batch_size)

    for category in categories:
        dataset = FairFaceDataset(root_dir, category, transform=predictor.transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        predictor.predict(dataloader)

