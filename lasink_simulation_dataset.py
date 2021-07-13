import glob
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torch
import random
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True


class LasinkSimulation(Dataset):
    def __init__(self, folder, transform=None):
        self.transforms=transform

        self.imgs = glob.glob(folder+'//*')

    def __getitem__(self, index):
        
        img = Image.open(self.imgs[index])
        if self.transforms:
            img = self.transforms(img)

        return img,  0

    def __len__(self):
        return len(self.imgs)
