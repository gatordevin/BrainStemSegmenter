
from enum import Enum

class DatasetType(Enum):
    IMAGE_CLASSIFICATION = "image_classification"
    IMAGE_DETECTION = "Image_detection"
    IMAGE_SEGMENTATION = "image_segmentation"
    IMAGE_COUNTING = "image_counting"

class DatasetDescriptor():
    def __init__(self):
        self.__dataset_type : DatasetType = DatasetType.IMAGE_CLASSIFICATION
        