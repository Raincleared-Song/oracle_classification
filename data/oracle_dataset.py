import glob
from typing import Optional, Callable
from torchvision.transforms import transforms
import torch
from PIL import Image
import torch.utils.data as D
import logging

logging.getLogger('PIL').setLevel(logging.WARNING)


class OracleDataset(D.Dataset):
    def __init__(self,
                 data_path: str,
                 transform: Optional[Callable] = None):
        self.img_paths = glob.glob(data_path)
        assert len(self.img_paths) > 100
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, -1

    def __len__(self):
        return len(self.img_paths)


class SupervisedOracleDataset(D.Dataset):
    def __init__(self, imgs,
                 transform: Optional[Callable] = transforms.Compose([transforms.ToTensor()]),
                 padding=False, return_path=False, get_image_fun=None):
        """

        @param imgs: list of (img_path, y)
        @param transform: transform methods
        """
        self.transform = transform
        self.imgs = imgs
        self.y = [item[1] for item in self.imgs]
        self.padding = padding
        self.return_path = return_path
        self.get_image_fun = get_image_fun

    def __getitem__(self, index):
        item = self.imgs[index]
        if self.get_image_fun is None:
            img = Image.open(item[0]).convert("RGB")
        else:
            img = self.get_image_fun(item[0])
        if self.transform is not None:
            img = self.transform(img)

        res = (img, torch.tensor(item[1]), )
        if self.return_path:
            res = (*res, item[0])
        return res

    def __len__(self):
        return len(self.imgs)
