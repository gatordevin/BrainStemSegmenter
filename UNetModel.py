import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl

# define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.unet = None # This will be set to a premade UNet pytorch model training will be handled by remaing functions
        # In order to accomplish trasnfer learning model that is loaded in should already be pretrained or have a checkpoint loaded
        # Also possible you need to freeze the model so it isnt randomized at the beginning.

    def training_step(self, batch, batch_idx): #Used when running training steps
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.unet(x)
        loss = nn.functional.mse_loss(y_hat, y)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx): #Used when testing the model on its own
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.unet(x)
        test_loss = nn.functional.mse_loss(y_hat, y)
        self.log("test_loss", test_loss)

    def validation_step(self, batch, batch_idx): #Used for validation while training (Small portion of test dataset)
        # this is the validation loop
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self.unet(x)
        val_loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer