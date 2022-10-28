import json
from rising import loading
from rising.loading import Dataset
import torch
import os
from PIL import Image, ImageDraw

class SegDataset(Dataset):
    def __init__(self, train: bool, data_dir: str, training_percentage : float):
        files = os.listdir(data_dir)
        data_paths = []
        for file in files:
            if(file.endswith(".tif")):
                paths = (data_dir+file, data_dir+file.replace(".tif", "_roi.zip"))
                data_paths.append(paths)
            
            num_train_samples = int(len(data_paths) * training_percentage)

            # Split train data into training and validation,
            # since test data contains no ground truth
            if train:
                data = data_paths[:num_train_samples]
            else:
                data = data_paths[num_train_samples:]

            self.data = data
            self.data_dir = data_dir

    def __getitem__(self, item: int) -> dict:
        sample = self.data[item]
        img = Image.open(sample[0]).convert('L')

        label = Image.open(sample[1]).convert('L')

        return {'data': torch.from_numpy(img).float(),
                'label': torch.from_numpy(label).float()}

    def __len__(self) -> int:
        """
        Adds a length to the dataset

        Returns:
            int: dataset's length
        """
        return len(self.data)