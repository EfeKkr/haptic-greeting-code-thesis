# Extracts frame-wise features from egocentric video data using a pretrained Vision Transformer (ViT-B/16).
# All frames will be resized, normalised, and sent to the ViT model. The embedding of [CLS] tokens is retained only. 
# The processing result is stored in [T, 768] .npy files located in the vit_features/ directory.

import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel
from torchvision import transforms
from PIL import Image

input_dir = "processed_data"          
output_dir = "vit_features"           
os.makedirs(output_dir, exist_ok=True)

# Load pretrained ViT model 
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
model.eval().cpu()

# Frame transform (normalize + resize to 224x224) 
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
])

for fname in tqdm(os.listdir(input_dir)):
    if not fname.endswith(".npy"):
        continue

    video_path = os.path.join(input_dir, fname)
    video = np.load(video_path)  # [T, H, W, C]
    features = []

    for frame in video:
        frame_tensor = transform(frame).unsqueeze(0) 
        with torch.no_grad():
            outputs = model(pixel_values=frame_tensor)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  
        features.append(cls_embedding.squeeze(0).numpy())      

    features = np.stack(features)  # [T, 768]
    out_path = os.path.join(output_dir, fname.replace(".npy", "_vit.npy"))
    np.save(out_path, features)

print("✅ Done extracting ViT features to:", output_dir)