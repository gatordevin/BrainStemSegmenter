from Datasets import CountceptionPickleDataset
from matplotlib import pyplot as plt
from Models import CountCeptionModel
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split

if __name__ == '__main__':
    dataset = CountceptionPickleDataset("objectOriented/MBM-dataset.pkl")

    train_dataset, val_dataset, test_dataset = random_split(dataset, [15,15,14])

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)

    model = CountCeptionModel()

    # training
    trainer = pl.Trainer(profiler="simple", accelerator='gpu', devices=1, precision=16, limit_train_batches=0.5, log_every_n_steps=1, max_epochs=1000)
    trainer.fit(model, train_loader, val_loader)