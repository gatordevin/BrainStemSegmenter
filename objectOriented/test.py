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
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = CountCeptionModel()
    model = model.load_from_checkpoint("lightning_logs/version_14/checkpoints/epoch=999-step=4000.ckpt")
    # model.eval()

    for idx, data in enumerate(test_loader):
        img = data[0]
        target = data[1]
        count = data[2]
        loss = model.training_step(data,idx)
        print(loss)
        pred = model.forward(img)
        f, axarr = plt.subplots(2,1) 
        axarr[0].imshow(target[0].detach().permute(1, 2, 0))
        axarr[1].imshow(pred[0].detach().permute(1, 2, 0))
        plt.show()
    # training

    # model.load_from_checkpoint(
    #     checkpoint_path="lightning_logs/version_14/checkpoints/epoch=999-step=4000.ckpt",
    #     hparams_file="lightning_logs/version_14/hparams.yaml",
    #     map_location=None,
    # )
    # trainer = pl.Trainer(accelerator='gpu', devices=1, precision=16, limit_train_batches=0.5, max_epochs=1000)
    # trainer.validate(model,val_loader,"lightning_logs/version_14/checkpoints/epoch=999-step=4000.ckpt")