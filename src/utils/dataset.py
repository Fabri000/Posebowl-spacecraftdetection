import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset

from utils.utility import get_labels, convert_labels_to_pixel

class SpacecraftDataset(Dataset):
    def __init__(self,set:str, base_labels_path:str="dataset/labels", base_image_path:str="dataset/images", data_path:str="/home/fabri/PoseBowl/SpacecraftDetection"):

        self.labels_df = convert_labels_to_pixel(get_labels(f"{base_labels_path}/{set}"))
        self.imgs_path = f"{data_path}/{base_image_path}/{set}"
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        
        e = self.labels_df.iloc[idx]
        
        img_pth = f"{self.imgs_path}/{e['image_id']}.jpg"
        img = cv2.imread(img_pth)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        boxes = torch.tensor([[e['x_min'], e['y_min'], e['x_max'], e['y_max']]], dtype=torch.int64)
        labels = torch.tensor([int(e['class'])], dtype=torch.int64)
        img_id = torch.tensor(idx, dtype=torch.int64)
        area = (e['x_max'] - e['x_min']) * (e['y_max'] - e['y_min'])

        target = {'boxes': boxes, 'labels': labels, 'image_id': img_id, 'area': area}

        return torch.tensor(img).permute(2, 0, 1), target
