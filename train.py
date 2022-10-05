import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from model import LitAutoEncoder
import torch

# init the autoencoder
autoencoder = LitAutoEncoder()

transform = ToTensor()
train_set = MNIST(root="MNIST", download=True, train=True, transform=transform)
test_set = MNIST(root="MNIST", download=True, train=False, transform=transform)

# use 20% of training data for validation
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = utils.data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

train_loader = utils.data.DataLoader(train_set)
valid_loader = utils.data.DataLoader(valid_set)
test_loader = utils.data.DataLoader(test_set)

trainer = pl.Trainer(limit_train_batches=100, max_epochs=20)
trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=valid_loader)

trainer.test(model=autoencoder, dataloaders=test_loader)