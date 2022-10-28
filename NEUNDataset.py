import json
from rising import loading
from rising.loading import Dataset
import torch
import os
from PIL import Image, ImageDraw
import numpy as np
class Dataset(Dataset):
    def __init__(self, train: bool, data_dir: str):
        """
        Args:
            train: whether to use the training or the validation split
            data_dir: directory containing the data
        """
        files = os.listdir(data_dir)
        file = iter(files)
        file_path_pairs = list(zip(file, file))
        files = []
        for file_path_pair in file_path_pairs:
            files.append({"image" : data_dir+"/"+file_path_pair[0], "label" : data_dir+"/"+file_path_pair[1]})

        num_train_samples = int(len(files) * 0.9)
        print(files)
        # Split train data into training and validation,
        # since test data contains no ground truth
        if train:
            data = files[:num_train_samples]
        else:
            data = files[num_train_samples:]

        self.data = data
        self.data_dir = data_dir

    def __getitem__(self, item: int) -> dict:
        """
        Loads and Returns a single sample

        Args:
            item: index specifying which item to load

        Returns:
            dict: the loaded sample
        """
        sample = self.data[item]
        img = Image.open(sample["image"])
        img = np.array(img)
        img = np.stack((img,)*3, axis=0)
        # add channel dim if necesary
        print(img.shape)
        print(img.ndim)
        if img.ndim == 3:
            img = img[None]

        label = Image.open(sample["label"])
        label = np.array(label)
        print(label.shape)
        # convert multiclass to binary task by combining all positives
        # label = label > 0

        # add channel dim if necessary
        if label.ndim == 3:
            label = label[None]
        return {'data': torch.from_numpy(img).float(),
                'label': torch.from_numpy(label).float()}

    def __len__(self) -> int:
        """
        Adds a length to the dataset

        Returns:
            int: dataset's length
        """
        return len(self.data)