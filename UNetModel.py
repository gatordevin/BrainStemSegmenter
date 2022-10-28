import pytorch_lightning as pl
import torch
import segmentation_models_pytorch as smp
from rising.transforms import NormZeroMeanUnitStd
from rising.loading import DataLoader
from rising.transforms.affine import BaseAffine
import random
from typing import Optional, Sequence, Union
from NEUNDataset import Dataset
from rising.transforms import Compose, ResizeNative, Scale
from torchvision import transforms
import numpy as np

def common_per_sample_trafos():
        return Compose(ResizeNative(size=(3, 640, 640), keys=('data',), mode='nearest'),
                       ResizeNative(size=(624), keys=('label',), mode='nearest'),)

def binary_dice_coefficient(pred: torch.Tensor, gt: torch.Tensor,
                            thresh: float = 0.5, smooth: float = 1e-7) -> torch.Tensor:
    """
    computes the dice coefficient for a binary segmentation task

    Args:
        pred: predicted segmentation (of shape Nx(Dx)HxW)
        gt: target segmentation (of shape NxCx(Dx)HxW)
        thresh: segmentation threshold
        smooth: smoothing value to avoid division by zero

    Returns:
        torch.Tensor: dice score
    """

    assert pred.shape == gt.shape

    pred_bool = pred > thresh

    intersec = (pred_bool * gt).float()
    return 2 * intersec.sum() / (pred_bool.float().sum()
                                 + gt.float().sum() + smooth)

class RandomAffine(BaseAffine):
    """Base Affine with random parameters for scale, rotation and translation"""
    def __init__(self, scale_range: Optional[tuple] = None,
                 rotation_range: Optional[tuple] = None,
                 translation_range: Optional[tuple] = None,
                 degree: bool = True,
                 image_transform: bool = True,
                 keys: Sequence = ('data',),
                 grad: bool = False,
                 output_size: Optional[tuple] = None,
                 adjust_size: bool = False,
                 interpolation_mode: str = 'nearest',
                 padding_mode: str = 'zeros',
                 align_corners: bool = False,
                 reverse_order: bool = False,
                 **kwargs,):

        """
        Args:
            scale_range: tuple containing minimum and maximum values for scale.
                Actual values will be sampled from uniform distribution with these
                constraints.
            rotation_range: tuple containing minimum and maximum values for rotation.
                Actual values will be sampled from uniform distribution with these
                constraints.
            translation_range: tuple containing minimum and maximum values for translation.
                Actual values will be sampled from uniform distribution with these
                constraints.
            keys: keys which should be augmented
            grad: enable gradient computation inside transformation
            degree: whether the given rotation(s) are in degrees.
                Only valid for rotation parameters, which aren't passed
                as full transformation matrix.
            output_size: if given, this will be the resulting image size.
                Defaults to ``None``
            adjust_size: if True, the resulting image size will be
                calculated dynamically to ensure that the whole image fits.
            interpolation_mode: interpolation mode to calculate output values
                ``'bilinear'`` | ``'nearest'``. Default: ``'bilinear'``
            padding_mode: padding mode for outside grid values
                ``'zeros'`` | ``'border'`` | ``'reflection'``.
                Default: ``'zeros'``
            align_corners: Geometrically, we consider the pixels of the
                input as squares rather than points. If set to True,
                the extrema (-1 and 1)  are considered as referring to the
                center points of the input’s corner pixels. If set to False,
                they are instead considered as referring to the corner points
                of the input’s corner pixels, making the sampling more
                resolution agnostic.
            reverse_order: reverses the coordinate order of the
                transformation to conform to the pytorch convention:
                transformation params order [W,H(,D)] and
                batch order [(D,)H,W]
            **kwargs: additional keyword arguments passed to the
                affine transf
        """
        super().__init__(scale=None, rotation=None, translation=None,
                         degree=degree,
                         image_transform=image_transform,
                         keys=keys,
                         grad=grad,
                         output_size=output_size,
                         adjust_size=adjust_size,
                         interpolation_mode=interpolation_mode,
                         padding_mode=padding_mode,
                         align_corners=align_corners,
                         reverse_order=reverse_order,
                         **kwargs)

        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.translation_range = translation_range

    def assemble_matrix(self, **data) -> torch.Tensor:
        """
        Samples Parameters for scale, rotation and translation
        before actual matrix assembly.

        Args:
            **data: dictionary containing a batch

        Returns:
            torch.Tensor: assembled affine matrix
        """
        ndim = data[self.keys[0]].ndim - 2

        if self.scale_range is not None:
            self.scale = [random.uniform(*self.scale_range) for _ in range(ndim)]

        if self.translation_range is not None:
            self.translation = [random.uniform(*self.translation_range) for _ in range(ndim)]

        if self.rotation_range is not None:
            if ndim == 3:
                self.rotation = [random.uniform(*self.rotation_range) for _ in range(ndim)]
            elif ndim == 1:
                self.rotation = random.uniform(*self.rotation_range)

        return super().assemble_matrix(**data)

class SoftDiceLoss(torch.nn.Module):
    """Soft Dice Loss"""
    def __init__(self, square_nom: bool = False,
                 square_denom: bool = False,
                 weight: Optional[Union[Sequence, torch.Tensor]] = None,
                 smooth: float = 1.):
        """
        Args:
            square_nom: whether to square the nominator
            square_denom: whether to square the denominator
            weight: additional weighting of individual classes
            smooth: smoothing for nominator and denominator

        """
        super().__init__()
        self.square_nom = square_nom
        self.square_denom = square_denom

        self.smooth = smooth

        if weight is not None:
            if not isinstance(weight, torch.Tensor):
                weight = torch.tensor(weight)

            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes SoftDice Loss

        Args:
            predictions: the predictions obtained by the network
            targets: the targets (ground truth) for the :attr:`predictions`

        Returns:
            torch.Tensor: the computed loss value
        """
        # number of classes for onehot
        n_classes = predictions.shape[1]
        with torch.no_grad():
            targets_onehot = rising.transforms.functional.channel.one_hot_batch(
                targets.unsqueeze(1), num_classes=n_classes)
        # sum over spatial dimensions
        dims = tuple(range(2, predictions.dim()))

        # compute nominator
        if self.square_nom:
            nom = torch.sum((predictions * targets_onehot.float()) ** 2, dim=dims)
        else:
            nom = torch.sum(predictions * targets_onehot.float(), dim=dims)
        nom = 2 * nom + self.smooth

        # compute denominator
        if self.square_denom:
            i_sum = torch.sum(predictions ** 2, dim=dims)
            t_sum = torch.sum(targets_onehot ** 2, dim=dims)
        else:
            i_sum = torch.sum(predictions, dim=dims)
            t_sum = torch.sum(targets_onehot, dim=dims)

        denom = i_sum + t_sum.float() + self.smooth

        # compute loss
        frac = nom / denom

        # apply weight for individual classesproperly
        if self.weight is not None:
            frac = self.weight * frac

        # average over classes
        frac = - torch.mean(frac, dim=1)

        return frac

class Unet(pl.LightningModule):
    """Simple U-Net without training logic"""
    def __init__(self, hparams: dict):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=hparams.get('in_channels', 1),                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=hparams.get('num_classes', 2),                      # model output channels (number of classes in your dataset)
        )
        
        # self.hparams = hparams

        self.dice_loss = SoftDiceLoss(weight=[0., 1.])
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        mask = self.model(input_tensor) 
        return mask

    def train_dataloader(self) -> DataLoader:
        """
        Specifies the train dataloader

        Returns:
            DataLoader: the train dataloader
        """
        # construct dataset
        dataset = Dataset(train=True, data_dir="processed/dataset_2")

        # specify batch transforms
        batch_transforms = Compose([
            RandomAffine(scale_range=(self.hparams.get('min_scale', 0.9), self.hparams.get('max_scale', 1.1)),
                         rotation_range=(self.hparams.get('min_rotation', -10), self.hparams.get('max_rotation', 10)),
                        keys=('data', 'label')),
            NormZeroMeanUnitStd(keys=('data',))
        ])

        # construct loader
        dataloader = DataLoader(dataset,
                                batch_size=self.hparams.get('batch_size', 1),
                                batch_transforms=batch_transforms,
                                shuffle=True,
                                sample_transforms=common_per_sample_trafos(),
                                pseudo_batch_dim=True,
                                num_workers=self.hparams.get('num_workers', 4))
        return dataloader

    def val_dataloader(self) -> DataLoader:
        # construct dataset
        dataset = Dataset(train=False, data_dir="processed/dataset_2")

        # specify batch transforms (no augmentation here)
        batch_transforms = NormZeroMeanUnitStd(keys=('data',))

        # construct loader
        dataloader = DataLoader(dataset,
                                batch_size=self.hparams.get('batch_size', 1),
                                batch_transforms=batch_transforms,
                                shuffle=False,
                                sample_transforms=common_per_sample_trafos(),
                                pseudo_batch_dim=True,
                                num_workers=self.hparams.get('num_workers', 4))

        return dataloader

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configures the optimier to use for training

        Returns:
            torch.optim.Optimier: the optimizer for updating the model's parameters
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.get('learning_rate', 1e-3))

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Defines the training logic

        Args:
            batch: contains the data (inputs and ground truth)
            batch_idx: the number of the current batch

        Returns:
            dict: the current loss value
        """
        x, y = batch['data'], batch['label']

        # remove channel dim from gt (was necessary for augmentation)
        y = y[:, 0].long()

        # obtain predictions
        pred = self(x)
        softmaxed_pred = torch.nn.functional.softmax(pred, dim=1)

        # Calculate losses
        ce_loss = self.ce_loss(pred, y)
        dice_loss = self.dice_loss(softmaxed_pred, y)
        total_loss = (ce_loss + dice_loss) / 2

        # calculate dice coefficient
        dice_coeff = binary_dice_coefficient(torch.argmax(softmaxed_pred, dim=1), y)

        # log values
        self.logger.experiment.add_scalar('Train/DiceCoeff', dice_coeff)
        self.logger.experiment.add_scalar('Train/CE', ce_loss)
        self.logger.experiment.add_scalar('Train/SoftDiceLoss', dice_loss)
        self.logger.experiment.add_scalar('Train/TotalLoss', total_loss)

        return {'loss': total_loss}

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Defines the validation logic

        Args:
            batch: contains the data (inputs and ground truth)
            batch_idx: the number of the current batch

        Returns:
            dict: the current loss and metric values
        """
        x, y = batch['data'], batch['label']

        # remove channel dim from gt (was necessary for augmentation)
        y = y[:, 0].long()

        # obtain predictions
        x = np.squeeze(x, axis=0)
        pred = self(x)
        softmaxed_pred = torch.nn.functional.softmax(pred, dim=1)

        # calculate losses
        ce_loss = self.ce_loss(pred, y)
        dice_loss = self.dice_loss(softmaxed_pred, y)
        total_loss = (ce_loss + dice_loss) / 2

        # calculate dice coefficient
        dice_coeff = binary_dice_coefficient(torch.argmax(softmaxed_pred, dim=1), y)

        # log values
        self.logger.experiment.add_scalar('Val/DiceCoeff', dice_coeff)
        self.logger.experiment.add_scalar('Val/CE', ce_loss)
        self.logger.experiment.add_scalar('Val/SoftDiceLoss', dice_loss)
        self.logger.experiment.add_scalar('Val/TotalLoss', total_loss)

        return {'val_loss': total_loss, 'dice': dice_coeff}

    def validation_epoch_end(self, outputs: list) -> dict:
        """Aggregates data from each validation step

        Args:
            outputs: the returned values from each validation step

        Returns:
            dict: the aggregated outputs
        """
        mean_outputs = {}
        for k in outputs[0].keys():
            mean_outputs[k] = torch.stack([x[k] for x in outputs]).mean()

        # tqdm.write('Dice: \t%.3f' % mean_outputs['dice'].item())
        return mean_outputs