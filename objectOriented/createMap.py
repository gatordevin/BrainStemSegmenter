from Datasets import CountceptionPickleDataset, CountceptionRawDataset
from matplotlib import pyplot as plt
from Models import CountCeptionModel
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split
import numpy as np
from skimage.io import imread
import torchvision.transforms as transforms
import torch
if __name__ == '__main__':
    # dataset = CountceptionPickleDataset("objectOriented/MBM-dataset.pkl")
    # dataset = CountceptionRawDataset("C:/Users/gator/FullerLab/BrainStemSegmenter/Data_10-28-2022/cropped")

    # # for data in dataset:
    # #     f, axarr = plt.subplots(2,1)
    # #     print(data[0].shape)
    # #     print(data[1][0].shape)
    # #     axarr[0].imshow(data[0].detach().permute(1, 2, 0))
    # #     print(data[1][0].dtype)
    # #     axarr[1].imshow(data[1][0].detach())
    # #     plt.show()
    # train_dataset, val_dataset, test_dataset = random_split(dataset, [174,20,20])

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = CountCeptionModel()
    model = model.load_from_checkpoint("C:/Users/gator/FullerLab/BrainStemSegmenter/lightning_logs/version_21/checkpoints/epoch=972-step=41839.ckpt")
    model.eval()
    # model = model.to(torch.device("cuda"))
    image_path = "C:/Users/gator/FullerLab/BrainStemSegmenter/Data_10-28-2022/1sAc1r2 PM NEUN.tif"
    image = imread(image_path)
    tensor_img = [transforms.ToTensor()(image)]
    # tensor_img = torch.stack(tensor_img).cuda()
    # pred = model.forward(tensor_img).cpu()
    tensor_img = torch.stack(tensor_img)
    pred = model.forward(tensor_img)
    pred = pred[0].detach().permute(1,2,0)
    f, axarr = plt.subplots(2,1)
    axarr[0].imshow(image)
    axarr[1].imshow(pred)
    plt.show()

    # for idx, data in enumerate(test_loader):
    #     img = data[0]
    #     target = data[1]
    #     count = data[2]
    #     loss = model.training_step(data,idx)
    #     print(loss)
    #     pred = model.forward(img)
    #     f, axarr = plt.subplots(1,1) 
    #     # axarr[0].imshow(target[0].detach().permute(1, 2, 0))
    #     print(img.shape)
    #     axarr.imshow(np.pad(img[0].detach().permute(1,2,0)[:,:,0],16, "constant"))
    #     axarr.imshow(pred[0].detach().permute(1, 2, 0),alpha=0.3)
    #     plt.savefig("image_"+str(idx)+".png")
    #     plt.show()


    # training

    # model.load_from_checkpoint(
    #     checkpoint_path="lightning_logs/version_14/checkpoints/epoch=999-step=4000.ckpt",
    #     hparams_file="lightning_logs/version_14/hparams.yaml",
    #     map_location=None,
    # )
    # trainer = pl.Trainer(accelerator='gpu', devices=1, precision=16, limit_train_batches=0.5, max_epochs=1000)
    # trainer.validate(model,val_loader,"lightning_logs/version_14/checkpoints/epoch=999-step=4000.ckpt")