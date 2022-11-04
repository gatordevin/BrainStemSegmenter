from Datasets import CountceptionPickleDataset, CountceptionRawDataset
from matplotlib import pyplot as plt
from Models import CountCeptionModel
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split
import numpy as np
from skimage.io import imread
import torchvision.transforms as transforms
import torch
import os
from PIL import Image
from torch.utils.data import Dataset
class CountceptionDataset(Dataset):
    def __init__(self, data_dir, train=False, transform=transforms.Compose([transforms.ToTensor()]), target_transform=transforms.Compose([transforms.ToTensor()])) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        data = []
        for image_name in os.listdir(self.data_dir):
            if("label" in image_name):
                input_name = image_name.replace("_label", "")
                target_name = image_name
                input_path = self.data_dir + "/" + input_name
                target_path = self.data_dir + "/" + target_name
                input_im = imread(input_path)
                target_im = Image.open(target_path)
                target_im = np.asarray(target_im, dtype=np.float32)
                # print("Dtype " + str(target_im.dtype))
                count = image_name.split("_")[-2]
                data.append([input_path, target_path, count])
        self.__dataset_images = np.asarray([d[0] for d in data])
        self.__dataset_heatmaps = np.asarray([d[1] for d in data])
        self.__dataset_counts = np.asarray([d[2] for d in data])
        self._data_pairs = list(zip(self.__dataset_images, list(zip(self.__dataset_heatmaps, self.__dataset_counts))))

        if(train):
            self._data_pairs = self._data_pairs[:int(len(self._data_pairs)*0.9)]
        else:
            self._data_pairs = self._data_pairs[int(len(self._data_pairs)*0.9):]

    def __getitem__(self, item: int):
        input_img = imread(self._data_pairs[item][0])
        target_img = Image.open(self._data_pairs[item][1][0])
        target_img = np.asarray(target_img, dtype=np.float32)
        if self.transform:
            input_img = self.transform(input_img)
        if self.target_transform:
            target_img = self.target_transform(target_img)
        return [input_img, target_img, self._data_pairs[item][1][1]]

    def __len__(self):
        return len(self._data_pairs)

import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from skimage.io import imread
from typing import Optional

class CountceptionDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        data = []
        for image_name in os.listdir(self.data_dir):
            if("label" in image_name):
                input_name = image_name.replace("_label", "")
                target_name = image_name
                input_path = self.data_dir + "/" + input_name
                target_path = self.data_dir + "/" + target_name
                input_im = imread(input_path)
                target_im = Image.open(target_path)
                target_im = np.asarray(target_im, dtype=np.float32)
                count = image_name.split("_")[-2]
                data.append([input_im, target_im, [count]])
        self.__dataset_images = np.asarray([d[0] for d in data])
        self.__dataset_heatmaps = np.asarray([d[1] for d in data])
        self.__dataset_counts = np.asarray([d[2] for d in data])
        self._data_pairs = list(zip(self.__dataset_images, list(zip(self.__dataset_heatmaps, self.__dataset_counts))))

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            countception_full = CountceptionDataset(self.data_dir, train=True, transform=self.transform)
            self.countception_train, self.countception_val = random_split(countception_full, [int(len(countception_full)*0.9), len(countception_full)-int(len(countception_full)*0.9)])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.countception_test = CountceptionDataset(self.data_dir, train=False, transform=self.transform)

        if stage == "predict" or stage is None:
            self.countception_predict = CountceptionDataset(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.countception_train, batch_size=2, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.countception_val, batch_size=2, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.countception_test, batch_size=8, num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.countception_predict, batch_size=8, num_workers=4)

model = CountCeptionModel()
# data_module = CountceptionDataModule("C:/Users/gator/BrainStemSegmenter/Data_10-28-2022/cropped")
data_module = CountceptionDataModule("C:/Users/gator/OneDrive - University of Florida/10x images for quantification/Manual Counts/cropped")

trainer = pl.Trainer(
    auto_lr_find=True, 
    profiler="simple", 
    accelerator='cpu', 
    devices=1, 
    precision=16, 
    limit_train_batches=0.5, 
    log_every_n_steps=1, 
    max_epochs=1000,
    num_sanity_val_steps=0
)
if __name__ == '__main__':
    trainer.fit(model, datamodule=data_module)