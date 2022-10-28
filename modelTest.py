from NEUNDataset import Dataset
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer
import torch

from UNetModel import Unet

if __name__ == '__main__':
    early_stop_callback = EarlyStopping(monitor='dice', min_delta=0.001, patience=10, verbose=False, mode='max')
    dataset = Dataset(True, "processed/dataset_2/")


    if torch.cuda.is_available():
        gpus = 1
    else:
        gpus = None

    nb_epochs = 50
    num_start_filts = 16
    num_workers = 4

    model = Unet({'in_channels': 1, 'num_classes': 3})

    trainer = Trainer(limit_train_batches=100, max_epochs=20)
    trainer.fit(model)