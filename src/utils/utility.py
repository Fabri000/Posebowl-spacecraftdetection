import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2,FasterRCNN_ResNet50_FPN_V2_Weights

def get_labels(path:str, base_path:str="/home/fabri/PoseBowl/SpacecraftDetection")-> pd.DataFrame:
    '''Returns a dataframe with the labels of the dataset expressed in yolo format.
    
    Args:
        path (str): The path to the labels folder, relative to the base_path.
        base_path (str): The base path to the dataset. Default is "/home/fabri/PoseBowl/SpacecraftDetection/dataset/labels".
    '''
    path = f"{base_path}/{path}"

    train_labels = os.listdir(path)

    labels = []
    for f in train_labels:
        with open(f"{path}/{f}") as file:
            s = file.read().rstrip()
            tmp = {}
            tmp['class'],tmp['x_center'],tmp['y_center'],tmp['width'],tmp['height'] = s.split(" ")
            tmp['class'] = int(tmp['class']) + 1 # 0 is associatd with background even though it is not detected in the dataset.
            tmp['image_id'] = f.split(".")[0]
            labels.append(tmp)

    labels_df = pd.DataFrame(labels)
    return labels_df

def convert_labels_to_pixel(labels_df:pd.DataFrame, image_width:int=1280, image_height:int=1024)-> pd.DataFrame:
    '''Converts the labels from yolo format to pixel format.
    
    Args:
        labels_df (pd.DataFrame): The dataframe with the labels in yolo format.
        image_width (int): The width of the images. Default is 1280.
        image_height (int): The height of the images. Default is 1024.
    '''
    converted_df = pd.DataFrame(columns=['x_min', 'y_min', 'x_max', 'y_max', 'image_id'])

    denormalized = {
        'x_center': labels_df['x_center'].astype(float) * image_width,
        'y_center': labels_df['y_center'].astype(float) * image_height,
        'width': labels_df['width'].astype(float) * image_width,
        'height': labels_df['height'].astype(float) * image_height
    }
    

    converted_df['x_min'] = (denormalized['x_center'] - (denormalized['width'] / 2)).astype(int)
    converted_df['y_min'] = (denormalized['y_center'] - (denormalized['height'] / 2)).astype(int)
    converted_df['x_max'] = (denormalized['x_center'] + (denormalized['width'] / 2)).astype(int)
    converted_df['y_max'] = (denormalized['y_center'] + (denormalized['height'] / 2)).astype(int)
    converted_df['image_id'] = labels_df['image_id']
    converted_df['class'] = labels_df['class']

    return converted_df

def convert_labels_to_yolo(labels_df:pd.DataFrame, image_width:int=1280, image_height:int=1024)-> pd.DataFrame:
    '''Converts the labels from pixel format to yolo format.
    
    Args:
        labels_df (pd.DataFrame): The dataframe with the labels in pixel format.'''
    converted_df = pd.DataFrame(columns=['x_center', 'y_center', 'width', 'height', 'image_id'])

    sizes= {
        'width': labels_df['x_max'] - labels_df['x_min'],
        'height': labels_df['y_max'] - labels_df['y_min']
    }

    locations = {
        'x_center': labels_df['x_min'] + (sizes['width'] / 2),
        'y_center': labels_df['y_min'] + (sizes['height'] / 2)
    }

    converted_df['x_center'] = locations['x_center'] / image_width
    converted_df['y_center'] = locations['y_center'] / image_height
    converted_df['width'] = sizes['width'] / image_width
    converted_df['height'] = sizes['height'] / image_height
    converted_df['image_id'] = labels_df['image_id']
    converted_df['class'] = labels_df['class']

    return converted_df

def get_model(model_name:str="fasterrcnn_resnet50"):
    '''Returns a pytorchvision model based on the model name.
    
    Args:
        model_name (str): The name of the model. Default is "fasterrcnn_resnet50".
    '''
    match model_name:
        case "fasterrcnn_resnet50":
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            model = fasterrcnn_resnet50_fpn_v2(weights=weights)
        case "rf_detr":
            weights = None
            
        case _:
            raise ValueError(f"Model {model_name} not supported.")
    return model
