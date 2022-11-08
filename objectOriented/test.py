from Datasets import CountceptionPickleDataset, CountceptionRawDataset
from matplotlib import pyplot as plt
from Models import CountCeptionModel
from notebookTrain import *
model = CountCeptionModel()
model = model.load_from_checkpoint("C:/Users/gator/FullerLab/BrainStemSegmenter/lightning_logs/version_35/checkpoints/epoch=999-step=10000.ckpt")
model.eval()
data_module = CountceptionDataModule("C:/Users/gator/OneDrive - University of Florida/10x images for quantification/Manual Counts/cropped")
data_module.setup("test")
loader = data_module.test_dataloader()
if __name__ == '__main__':
    for batch in loader:
        images = batch[0]
        labels = batch[1]
        counts = batch[2]
        for image, label, count in list(zip(images, labels, counts)):
            image_model = torch.stack([image])
            pred = model.forward(image_model)
            patch_size = 32
            ef = ((patch_size / 1) ** 2.0)
            pred_count = (pred.detach().numpy() / ef).sum(axis=(2, 3))
            image = image.detach().permute(1,2,0)
            pred = pred[0].detach().permute(1,2,0)
            label = label.detach().permute(1,2,0)
            f, axarr = plt.subplots(2,2)
            # plt.imshow(np.pad(image[:,:,1],int(patch_size/2), "constant"))
            axarr[0][0].imshow(label)
            axarr[1][0].imshow(pred)
            axarr[0][1].imshow(image)
            NEUN_image = image[:,:,1]*3
            axarr[1][1].imshow(np.stack([NEUN_image,NEUN_image,NEUN_image]).transpose(1,2,0))
            
            plt.title("Real count: " + count + " Pred count: " +str(pred_count))
            plt.savefig("Real coun " + count + " Pred count " +str(pred_count[0][0]) + " test plot.png", dpi=800)
            # plt.show()
        print(images.shape)
        print(labels.shape)
        print(counts)