import pandas as pd
import os

def get_labels(path:str, base_path:str="/home/fabri/PoseBowl/SpacecraftDetection/dataset/labels")-> pd.DataFrame:
    path = f"{base_path}/{path}"

    train_labels = os.listdir(path)

    labels = []
    for f in train_labels:
        with open(f"{path}/{f}") as file:
            s = file.read().rstrip()
            tmp = {}
            tmp['image_id'],tmp['xmin'],tmp['ymin'],tmp['xmax'],tmp['ymax'] = s.split(" ")
            labels.append(tmp)

    labels_df = pd.DataFrame(labels)
    return labels_df