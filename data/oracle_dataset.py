import os
from typing import Optional, Callable
from torchvision.transforms import transforms
import torch
from PIL import Image
import torch.utils.data as D
import logging
from utils import load_json

logging.getLogger('PIL').setLevel(logging.WARNING)


class OracleDataset(D.Dataset):

    src_to_idx = {'wzb': 0, 'chant': 1}
    idx_to_dir = ['/var/lib/shared_volume/home/linbiyuan/corpus/wenbian/labels_页数+序号+著录号+字形_校对版_060616/char',
                  '/data/private/songchenyang/hanzi_filter/handa']

    def __init__(self,
                 data_path: str,
                 transform: Optional[Callable] = None):
        self.indexes = {int(key): val for key, val in load_json(data_path).items()}
        cur_images = []
        for _, val in self.indexes.items():
            for img in val:
                cur_images.append((img['img'], OracleDataset.src_to_idx[img['src']]))
        self.img_paths = cur_images
        assert len(self.img_paths) > 100
        self.transform = transform

    def __getitem__(self, index):
        img, idx = self.img_paths[index]
        img = Image.open(os.path.join(OracleDataset.idx_to_dir[idx], img)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img, {'src': idx})
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
