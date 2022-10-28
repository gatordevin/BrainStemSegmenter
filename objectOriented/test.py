from Datasets import CountceptionDataset
from matplotlib import pyplot as plt
from Models import CountCeptionModel
from torch.utils.data import DataLoader
import pytorch_lightning as pl

dataset = CountceptionDataset("objectOriented/MBM-dataset.pkl")
train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

model = CountCeptionModel()

# training
trainer = pl.Trainer(gpus=1, precision=16, limit_train_batches=0.5)
trainer.fit(model, train_loader,)