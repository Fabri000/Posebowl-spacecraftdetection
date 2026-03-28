import os
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
    
class DatasetConverter2Coco(DatasetConverter):
    '''Converts a dataset from the original format to the COCO format.'''

    def __init__(self, original_dataset_path:str, converted_dataset_path:str):
        super().__init__(original_dataset_path, converted_dataset_path)
        self.images_path = f"{self.original_dataset_path}/images"
        self.annotations_path = f"{self.original_dataset_path}/annotations"

    def convert(self):
        '''Converts the dataset from the original format to the COCO format.'''
        if not os.path.exists(self.converted_dataset_path):
            os.makedirs(self.converted_dataset_path)
            os.makedirs(f"{self.converted_dataset_path}/images")
            os.makedirs(f"{self.converted_dataset_path}/annotations")

        

