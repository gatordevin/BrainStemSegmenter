import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
import torch.nn.init as init

class ConvBlock(nn.Module):
    def __init__(self, in_chan, out_chan, ksize=3, stride=1, pad=0, activation=nn.LeakyReLU()):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=ksize, stride=stride, padding=pad)
        self.activation = activation
        self.batch_norm = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        return self.activation(self.batch_norm(self.conv1(x)))


class SimpleBlock(nn.Module):
    def __init__(self, in_chan, out_chan_1x1, out_chan_3x3, activation=nn.LeakyReLU()):
        super(SimpleBlock, self).__init__()
        self.conv1 = ConvBlock(in_chan, out_chan_1x1, ksize=1, pad=0, activation=activation)
        self.conv2 = ConvBlock(in_chan, out_chan_3x3, ksize=3, pad=1, activation=activation)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(x)
        output = torch.cat([conv1_out, conv2_out], 1)
        return output

class ModelObject(pl.LightningModule):
    def __init__(self):
        super().__init__()

class CountCeptionModel(ModelObject):
    def __init__(self, inplanes=3, outplanes=1, use_logits=False, logits_per_output=12, debug=False, lr=0.001):
        super().__init__()
        self.lr = lr
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.activation = nn.LeakyReLU(0.01)
        self.final_activation = nn.LeakyReLU(0.01)
        self.patch_size = 32
        self.use_logits = use_logits
        self.logits_per_output = logits_per_output
        self.debug = debug

        torch.LongTensor()

        self.conv1 = ConvBlock(self.inplanes, 64, ksize=3, pad=self.patch_size, activation=self.activation)
        self.simple1 = SimpleBlock(64, 16, 16, activation=self.activation)
        self.simple2 = SimpleBlock(32, 16, 32, activation=self.activation)
        self.conv2 = ConvBlock(48, 16, ksize=14, activation=self.activation)
        self.simple3 = SimpleBlock(16, 112, 48, activation=self.activation)
        self.simple4 = SimpleBlock(160, 64, 32, activation=self.activation)
        self.simple5 = SimpleBlock(96, 40, 40, activation=self.activation)
        self.simple6 = SimpleBlock(80, 32, 96, activation=self.activation)
        self.conv3 = ConvBlock(128, 32, ksize=18, activation=self.activation)
        self.conv4 = ConvBlock(32, 64, ksize=1, activation=self.activation)
        self.conv5 = ConvBlock(64, 64, ksize=1, activation=self.activation)
        if use_logits:
            self.conv6 = nn.ModuleList([ConvBlock(
                64, logits_per_output, ksize=1, activation=self.final_activation) for _ in range(outplanes)])
        else:
            self.conv6 = ConvBlock(64, self.outplanes, ksize=1, activation=self.final_activation)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.xavier_uniform_(m.weight, gain=init.calculate_gain('leaky_relu', param=0.01))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.model = nn.Sequential(
            self.conv1,
            self.simple1,
            self.simple2,
            self.conv2,
            self.simple3,
            self.simple4,
            self.simple5,
            self.simple6,
            self.conv3,
            self.conv4,
            self.conv5
        )
        if self.use_logits:
            for block in self.conv6:
                self.model = nn.Sequential(
                    self.model,
                    block
                )
        else:
            self.model = nn.Sequential(
                    self.model,
                    self.conv6
                )
        self.criterion = nn.L1Loss()
        
    def forward(self, x):
        pred = self.model(x)
        return pred

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        input, heatmap, count = train_batch
        pred = self.model(input)
        loss = self.criterion(pred, heatmap)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        tensorboard = self.logger.experiment
        input, heatmap, count = val_batch
        pred = self.model(input)
        loss = self.criterion(pred, heatmap)
        # tensorboard.add_image('target', heatmap[0], 0, dataformats='CHW')
        # tensorboard.add_image('pred', pred[0], 0, dataformats='CHW')
        # patch_size = 32
        # ef = ((patch_size / 1) ** 2.0)
        # for pred_i, count_i in zip(pred, count):
        #     output_count = (pred_i.cpu().detach().numpy() / ef).sum(axis=(1, 2))[0]
        #     # print(output_count)
        #     target_count = count_i.data.cpu().detach().numpy()[0]
        #     # print(target_count)
        #     count_diff = output_count-target_count
        #     self.log('val_count_diff', count_diff)
        # print(pred.shape)
        self.log('val_loss', loss)

    def test_step(self, val_batch, batch_idx):
        input, heatmap, count = val_batch
        pred = self.model(input)
        loss = self.criterion(pred, heatmap)
        self.log('test_loss', loss)
