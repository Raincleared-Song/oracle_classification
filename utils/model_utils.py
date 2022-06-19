import os
import yaml
import torch
import pickle
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.nn import functional as F


def save_checkpoint(state, is_best=False, filename='checkpoint.pth.tar', model_best='model_best.pth.tar'):
    print(f"saving checkpoint to {filename}")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, model_best)


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,), return_pred=False):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(float(correct_k.mul_(100.0)))
        if return_pred:
            res = (*res, pred)
        return res


def euclidean_dist(x, y):
    """
    Compute euclidean distance between two tensors
    """
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def cosine_dist(x, y):  # 效果不好
    return -torch.matmul(F.normalize(x), F.normalize(y).T)


def get_last_part(path):
    return path.strip("/").split("/")[-1]


def turn_img_white(img):
    if len(img[img > 250]) < img.size * 0.5:
        img = 255 - img
    return img


def preprocess(img):
    # cut images white edges.
    # detect the character in the image and crop the character part
    img = np.array(img)

    # turn image to white background
    img = turn_img_white(img)

    # # binarize
    img = np.uint8((img > 128) * 255)

    # crop to square
    left, right, up, down = 10000, -1, 10000, -1
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            if img[i, j] < 255:
                left = min(left, j)
                up = min(up, i)
                right = max(right, j)
                down = max(down, i)
    if not (left >= 0 and right >= 0 and up >= 0 and down >= 0):
        return Image.fromarray(img)
    if left >= right or up >= down:
        return Image.fromarray(img)
    vert_len = down - up
    hori_len = right - left
    width = max(vert_len, hori_len)
    new_img = np.ones((width, width)) * 255
    if vert_len > hori_len:
        hori_start = (vert_len - hori_len) // 2
        vert_start = 0
    else:
        vert_start = (hori_len - vert_len) // 2
        hori_start = 0
    new_img[vert_start:vert_start + vert_len, hori_start:hori_start + hori_len] = img[up:down, left:right]
    return Image.fromarray(new_img)


class Counter:
    def __init__(self):
        self.cnt = 0
        self.items = list()

    def add(self, item, cnt=1):
        self.items.append(item)
        self.cnt += cnt

    def __len__(self):
        return self.cnt

    def sum(self):
        return sum(self.items)

    def average(self):
        if self.cnt == 0:
            return 0
        return sum(self.items)/self.cnt

    def threshold_cnt(self, threshold=0.9):
        return sum([int(item > threshold) for item in self.items])


def get_feats(model, dataloader, device, cache_path=None):
    if not os.path.exists("./cache"):
        os.mkdir("./cache")
    if cache_path is not None:
        cache_path = "./cache" + "/" + cache_path + ".pkl"
    if cache_path is not None and os.path.exists(cache_path):
        feats, y2ind = pickle.load(open(cache_path, 'rb'))
        return feats.to(device), y2ind
    y2feats = {}
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            x, y = batch
            x, y = x.to(device), y
            protos = model(x)
            for i, proto in enumerate(protos):
                label = int(y[i])
                if y2feats.get(label) is None:
                    y2feats[label] = []
                y2feats[label].append(proto)
    ys = sorted(list(y2feats.keys()))
    y2ind = {}  # label 未必完全连续，存在没有任何样本的label，y2ind将labels映射到紧密的区间
    for ind, y in enumerate(list(y2feats.keys())):
        y2ind[y] = ind  # note: y2ind未必包含所有的y
    feats = torch.empty((len(y2ind), list(y2feats.values())[0][0].shape[0]), device=device, dtype=torch.float)
    for y in ys:
        protos = y2feats[y]
        feats[y2ind[y]] = sum(protos) / len(protos)
    if cache_path is not None:
        pickle.dump((feats.cpu(), y2ind), open(cache_path, 'wb'))
    return feats, y2ind
