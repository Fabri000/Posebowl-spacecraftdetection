import os
import shutil
import json
from abc import ABC


class DatasetConverter(ABC):
    '''Abstract class for converting a dataset from one format to another.'''
    def __init__(self, original_dataset_path:str, converted_dataset_path:str):
        self.original_dataset_path = original_dataset_path
        if not os.path.exists(original_dataset_path):
            raise ValueError(f"Original dataset path {original_dataset_path} does not exist.")
        self.converted_dataset_path = converted_dataset_path
    
    def convert(self):
        '''Converts the dataset from the original format to the converted format.'''
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def _denormalize_label(self, x_center:float,y_center:float,width:float,height:float,image_width:int=1280,image_height:int=1024):
        '''Denormalizes the labels from the yolo format to the pixel format.'''
        x_center = int(x_center * image_width)
        y_center = int(y_center * image_height)
        width = int(width * image_width)
        height = int(height * image_height)
        return x_center, y_center, width, height

    def _center_to_border(x_center:int,y_center:int,width:int,height:int):
        '''Converts the labels from the center format to the left-top border format.'''
        x_min = x_center - (width / 2)
        y_min = y_center - (height / 2)
        return int(x_min),int(y_min)
    
    def _border_to_center(x_min:int,y_min:int,width:int,height:int):
        '''Converts the labels from the left-top border format to the center format.'''
        x_center = x_min + (width / 2)
        y_center = y_min + (height / 2)
        return int(x_center), int(y_center)
    
class DatasetConverter2Coco(DatasetConverter):
    '''Converts a dataset from the original format to the COCO format.'''

    def __init__(self, original_dataset_path:str, converted_dataset_path:str):
        super().__init__(original_dataset_path, converted_dataset_path)
        self.images_path = f"{self.original_dataset_path}/images"
        self.labels_path = f"{self.original_dataset_path}/labels"

    def convert(self):
        '''Converts the dataset from the original format to the COCO format.'''
        if not os.path.exists(self.converted_dataset_path):
            os.makedirs(self.converted_dataset_path)
            os.makedirs(f"{self.converted_dataset_path}/images")
            os.makedirs(f"{self.converted_dataset_path}/annotations")


        for entry in os.listdir(self.images_path):
            print(f"Processing {entry}...")
            original_image_path = os.path.join(self.images_path, entry)
            shutil.copytree(original_image_path, f"{self.converted_dataset_path}/images",dirs_exist_ok=True)

            self._convert_labels_to_coco(entry)

  
    def _convert_labels_to_coco(self, labels_set:str,images_width:int=1280,images_height:int=1024):
        '''Converts the labels from the original format to the COCO format.'''

        labels_file = f"{labels_set}_labels.json"
        
        json_dict = {
                "images": [],
                "annotations": [],
                "categories": [{"supercategory": "Spacecraft", "id": 1, "name": "spacecraft"}]
            }
        
        labels = f"{self.labels_path}/{labels_set}"
        
        files = [f for f in os.listdir(labels) if os.path.isfile(os.path.join(labels, f))]
        for entry in files:
            with open(f"{labels}/{entry}") as file:
                s = file.read().rstrip()
                class_id, x_center, y_center, width, height = s.split(" ")
                x_center, y_center, width, height = self._denormalize_label(float(x_center), float(y_center), float(width), float(height))
                x_min,y_min = DatasetConverter._center_to_border(x_center, y_center, width, height)

                image_name = entry.split(".")[0]
                image_id = len(json_dict["images"]) + 1

                json_dict["images"].append({
                    "id": image_id,
                    "file_name": f"{image_name}.jpg",
                    "width": images_width,
                    "height": images_height
                })

                json_dict["annotations"].append({
                    "id": len(json_dict["annotations"]) + 1,
                    "image_id": image_id,
                    "category_id": int(class_id) + 1,
                    "bbox": [x_min,y_min,width,height],
                    "area": float(width) * float(height),
                    "iscrowd": 0
                })

                file.close()

        with open(f"{self.converted_dataset_path}/annotations/{labels_file}", "w") as f:
            json.dump(json_dict, f)