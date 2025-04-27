#Dataset

import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
from config import resize_x, resize_y, input_channels, group_size, epsilon, data_dir, csv_path, batchsize

def get_transform():
    return transforms.Compose([
        transforms.Resize((resize_x, resize_y)),  # Ensure image is 128x128.
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0457, 0.0402, 0.0299], std=[0.0828, 0.0696, 0.0595])
    ])

group_size = group_size
ep = epsilon

def normalize_groupwise(label_vector):
    start = 0
    normalized = []
    for size in group_size:
        group = np.array(label_vector[start:start+size])
        denom = np.sum(group) + ep
        norm_group = group / denom
        normalized.extend(norm_group.tolist())
        start += size
    return normalized

class GalaxyZooDataset(Dataset):
    def __init__(self, images_dir, csv_file, transform=None):
        self.images_dir = images_dir
        self.transform = transform or get_transform()

        # Automatically detect delimiter and read the CSV.
        # Using engine="python" with sep=None lets pandas try to infer the delimiter.
        self.data = pd.read_csv(csv_file, sep=None, engine="python")
        
        # Ensure that the header row is used correctly.
        # The CSV should have a header with "GalaxyID" and the remaining columns are class scores.
        self.prob_columns = [col for col in self.data.columns if col.strip() != "GalaxyID"]
        
        # Build a dictionary mapping GalaxyID to normalized probability vector.
        self.labels_dict = {}
        for _, row in self.data.iterrows():
            try:
                galaxy_id = int(row["GalaxyID"])
            except Exception as e:
                print("Error converting GalaxyID:", e)
                continue
            
            try:
                # Explicitly convert class score values to float.
                scores = [float(row[col]) for col in self.prob_columns]
            except Exception as e:
                print(f"Error converting scores for GalaxyID {galaxy_id}: {e}")
                scores = [0.0] * len(self.prob_columns)
            normalized_scores = normalize_groupwise(scores)
            self.labels_dict[galaxy_id] = torch.tensor(normalized_scores, dtype=torch.float)
        
        # List image paths in the images directory.
        self.image_paths = sorted(glob.glob(os.path.join(self.images_dir, "*.jpg")))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        base = os.path.basename(img_path)

        try:
            # Strip extension and extract GalaxyID
            galaxy_id_str = os.path.splitext(base)[0].split('_')[0]
            galaxy_id = int(galaxy_id_str)
        except Exception as e:
            print("Error extracting GalaxyID from filename:", e)
            galaxy_id = None

        target = self.labels_dict.get(galaxy_id, torch.zeros(len(self.prob_columns)))
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, target
    
class GalaxyZooDataloader(DataLoader):
    def __init__(self, images_dir, csv_file, batch_size):
        self.dataset = GalaxyZooDataset(images_dir=images_dir, csv_file=csv_file, transform=get_transform())
        super().__init__(self.dataset, batch_size=batch_size)




