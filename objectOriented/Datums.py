from PIL import Image 
import PIL
import numpy as np
import torchvision.transforms as transforms
import torch

class DataOject():
    def __init__(self) -> None:
        self._data = None

    def get_data(self):
        return self._data
    def _set_data(self, data):
        self._data = data

    def as_tensor(self):
        # print(self._data.dtype)
        tensor = transforms.ToTensor()(self._data)
        # print(tensor.dtype)
        return tensor

class ImageDataObject(DataOject):
    def __init__(self, image_path=None, image_array=None) -> None:
        super().__init__()
        self.__image_path = image_path
        self._set_data(image_array)

    def save(self):
        if(self.__image_path!=None):
            pil_img = Image.fromarray(self._data)
            pil_img = pil_img.save(self.__image_path)

    def load(self):
        if(self.__image_path!=None):
            pil_img = Image.open(self.__image_path)
            self._set_data(np.array(pil_img))

    def _set_data(self, data):
        if(data is not None):
            if(len(data.shape)>2):
                if(data.shape[0]<4):
                    data = data.transpose(1,2,0)
            self._data = data

class NumberDataObject(DataOject):
    def __init__(self, number) -> None:
        super().__init__()
        self._set_data(number)

    def as_tensor(self):
        return np.array([self._data])