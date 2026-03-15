from torch.utils.data import Dataset
import pandas as pd

class SpacecraftDataset(Dataset):
    def __init__(self, labels_path:str="", base_image_path:str="dataset/images"):
        self.labels_df = pd.read_csv(labels_path)
        self.labels_df['label'] = pd.zeros(len(self.labels_df))
        self.base_path = base_image_path
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        pass