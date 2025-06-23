# Extracts deep features from egocentric video frames using a pretrained ResNet18 CNN.
# Frames in the form of .npy files are loaded, transformed (resized and normalized), they go through the CNN, then 
# saved as fixed size feature vectors representing each video stored as files 
# in the "features/" folder.

import os
import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from scripts.dataset import GestureDataset
from torch.utils.data import DataLoader

input_dir = "processed_data"
output_dir = "features"
os.makedirs(output_dir, exist_ok=True)
batch_size = 1  # process one video at a time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Pretrained CNN (ResNet18)
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
cnn = nn.Sequential(*list(resnet.children())[:-1])
cnn.eval().to(device)

#  Transform (applied to each frame)
video_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = GestureDataset(input_dir, transform=video_transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

for i, (video_tensor, _) in enumerate(tqdm(loader)):
    video_tensor = video_tensor.squeeze(0).to(device)  # [T, C, H, W]
    with torch.no_grad():
        features = cnn(video_tensor)  # [T, 512, 1, 1]
        features = features.view(features.size(0), -1).cpu().numpy()

    filename = dataset.data_files[i]
    save_path = os.path.join(output_dir, filename)
    np.save(save_path, features)