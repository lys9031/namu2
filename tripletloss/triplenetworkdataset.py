import PIL.ImageOps
from tripletloss.triplenetwork import *
import numpy as np
from PIL import Image


class TripletNetworkDataset(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True, train=True):
        self.is_train = train
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

        if self.is_train:
            self.index = np.array(list(range(len(imageFolderDataset))))
            self.labels = np.array(imageFolderDataset.targets)
            self.images = imageFolderDataset.samples
        else:
            self.index = np.array(list(range(len(imageFolderDataset))))
            self.labels = np.array(imageFolderDataset.targets)
            self.images = imageFolderDataset.samples

    def __getitem__(self, item):  # DataLoader shuffle=True... 0,1,2 random
        imgA_img = self.images[item][0]
        imgA_label = self.labels[item]
        # print('imgA_label:', imgA_label)

        if self.is_train:

            # ahchor가 아닌 것들 중에서 label 같은 것들의 index를 가지고 옮
            imgP_list = self.index[self.index != item][self.labels[self.index != item] == imgA_label]
            imgP_item = random.choice(imgP_list)
            imgP_img = self.images[imgP_item][0]

            imgN_list = self.index[self.index != item][self.labels[self.index != item] != imgA_label]
            imgN_item = random.choice(imgN_list)
            imgN_img = self.images[imgN_item][0]

            imgA = Image.open(imgA_img)
            imgP = Image.open(imgP_img)
            imgN = Image.open(imgN_img)
            imgA = imgA.convert("RGB")
            imgP = imgP.convert("RGB")
            imgN = imgN.convert("RGB")

            if self.should_invert:
                imgA = PIL.ImageOps.invert(imgA)
                imgP = PIL.ImageOps.invert(imgP)
                imgN = PIL.ImageOps.invert(imgN)
            if self.transform is not None:
                imgA = self.transform(imgA)
                imgP = self.transform(imgP)
                imgN = self.transform(imgN)

            return imgA, imgP, imgN, imgA_label
        else:
            imgA = Image.open(imgA_img)
            anchor_img = imgA.convert('RGB')

            if self.transform is not None:
                anchor_img = self.transform(anchor_img)
            return anchor_img, imgA_label
    def __len__(self):
        return len(self.imageFolderDataset.imgs)