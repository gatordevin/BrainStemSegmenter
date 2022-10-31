from cgi import test
from Datums import *
from typing import *
import pickle
import numpy as np
from torch.utils.data import Dataset
from math import ceil
import os
import scipy
from skimage.io import imread

class DatasetObject(Dataset):
    def __init__(self) -> None:
        self._data_pairs = [[]]
    
    def get_index(self, item: int):
        return self._data_pairs[item]

    def __getitem__(self, item: int):
        return self.get_index(item)

    def __len__(self):
        return len(self._data_pairs)

class COCODatasetObject(DatasetObject):
    def __init__(self) -> None:
        super().__init__()
        self.__coco_json = {}

    def __prepare_data(self):
        # Go through coco json and fill in self.__data_pairs
        self._data_pairs[0]

    def __getitem__(self, item: int):
        input, target = super().__getitem__(item)
        image = ImageDataObject(image_path=input)
        return image, target

class CountceptionPickleDataset(DatasetObject):
    def __init__(self, pickle_dataset_path) -> None:
        super().__init__()
        self.__pickle_dataset_path = pickle_dataset_path
        self.__prepare_data()
        
    def __prepare_data(self):
        self.__pickle_dataset = pickle.load(open(self.__pickle_dataset_path, "rb"))
        self.__dataset_images = np.asarray([ImageDataObject(image_array = d[0]) for d in self.__pickle_dataset])
        self.__dataset_heatmaps = np.asarray([ImageDataObject(image_array = d[1]) for d in self.__pickle_dataset])
        self.__dataset_counts = np.asarray([NumberDataObject(d[2][0]) for d in self.__pickle_dataset])
        self._data_pairs = list(zip(self.__dataset_images, list(zip(self.__dataset_heatmaps, self.__dataset_counts))))

    def __getitem__(self, item: int):
        image, (heatmap, count) = super().__getitem__(item)
        return image.as_tensor(), heatmap.as_tensor(), count.as_tensor()

class CountceptionRawDataset(DatasetObject):
    def __init__(self, data_folder) -> None:
        super().__init__()
        self.__data_folder_path = data_folder
        self.__prepare_data()
        
    def __prepare_data(self):
        data = []
        for image_name in os.listdir(self.__data_folder_path):
            if("label" in image_name):
                input_name = image_name.replace("_label", "")
                target_name = image_name
                input_path = self.__data_folder_path + "/" + input_name
                target_path = self.__data_folder_path + "/" + target_name
                input_im = imread(input_path)
                target_im = imread(target_path)
                count = image_name.split("_")[-2]
                data.append([input_im, target_im, count])
            self.__dataset_images = np.asarray([ImageDataObject(image_array = [0]) for d in data])
            self.__dataset_heatmaps = np.asarray([ImageDataObject(image_array = d[1]) for d in data])
            self.__dataset_counts = np.asarray([NumberDataObject(d[2]) for d in data])
            self._data_pairs = list(zip(self.__dataset_images, list(zip(self.__dataset_heatmaps, self.__dataset_counts))))

    def __getitem__(self, item: int):
        image, (heatmap, count) = super().__getitem__(item)
        return image.as_tensor(), heatmap.as_tensor(), count.as_tensor()
        