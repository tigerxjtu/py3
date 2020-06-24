import numpy as np
import pandas as pd
import gc

import warnings
warnings.filterwarnings('ignore')

import os
import glob
import os.path as osp
from PIL import Image

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils import data as D

path = r'c:/download/train'
ds_path=r'c:/download'

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

class AirbusDS(D.Dataset):
    """
    A customized data loader.
    """

    def __init__(self, root, masks, mode='train'):
        """ Intialize the dataset
        """
        self.filenames = []
        self.root = root
        self.transform = transforms.ToTensor()
        filenames = glob.glob(osp.join(path, '*.jpg'))
        for fn in filenames:
            self.filenames.append(fn)
        self.len = len(self.filenames)
        self.mode = mode
        if mode == 'train':
            self.masks = masks

    # You must override __getitem__ and __len__
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        image = Image.open(self.filenames[index])
        ImageId = self.filenames[index].split(os.sep)[-1]
        if self.mode == 'train':
            mask = self.get_mask(ImageId)
            # print(np.sum(mask))
            return self.transform(image), mask[np.newaxis, :, :]
        return self.transform(image)

    # You must override __getitem__ and __len__
    def get_mask(self, ImageId):
        img_masks = self.masks.loc[self.masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()
        # print(ImageId,img_masks)

        # Take the individual ship masks and create a single mask array for all ships
        all_masks = np.zeros((768, 768))
        if img_masks == [-1]:
            return all_masks
        for mask in img_masks:
            all_masks += rle_decode(mask)
        return all_masks

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len

# functions to show an image
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

masks = pd.read_csv(os.path.join(ds_path,'train_ship_segmentations.csv')).fillna(-1)
print(masks.head())

airimg = AirbusDS(path,masks)
# total images in set
print(airimg.len)

train_len = int(0.7*airimg.len)
valid_len = airimg.len - train_len
train, valid = D.random_split(airimg, lengths=[train_len, valid_len])

loader = D.DataLoader(train, batch_size=24, shuffle=True, num_workers=0)

# get some images
dataiter = iter(loader)
images, masks = dataiter.next()

# show images
plt.figure(figsize=(16,16))
plt.subplot(211)
imshow(torchvision.utils.make_grid(images))
plt.subplot(212)
imshow(torchvision.utils.make_grid(masks))
plt.show()

