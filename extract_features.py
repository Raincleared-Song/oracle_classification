import sys

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms
import glob
import random
from PIL import Image
from tqdm import tqdm
import json
import shutil
import os
from utils import preprocess, daystr

from data.oracle_dataset import SupervisedOracleDataset
from data.utils import get_oracle_paths, get_standard_paths, get_handa_classes_data, get_oracle_tokens, get_oracle_db_data, get_jiaguwenbian_data
from utils import device, get_last_part, get_feats

class SelfMap(nn.Module):
    def __init__(self):
        super(SelfMap, self).__init__()
    def forward(self, x):
        return x




def get_img(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)

def get_feats_from_paths(paths):
    feats = []
    with torch.no_grad():
        for path in tqdm(paths):
            img = get_img(path).to(device)
            feat = model(img).squeeze(0)
            feats.append(feat)
    feats = torch.stack(feats, 0)
    # feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats

# def get_feats(model, dataloader):
#     y2feats = {}
#     for x_batch, y_batch in tqdm(dataloader):
#         with torch.no_grad():
#             feats = model(x_batch.to(device))
#             for i, feat in enumerate(feats):
#                 label = int(y_batch[i])
#                 if y2feats.get(label) is None:
#                     y2feats[label] = []
#                 y2feats[label].append(feat)
#     mean_feats = torch.stack([sum(feats)/len(feats) for feats in y2feats.values()])
#     ind2y = {}
#     ys = sorted(list(y2feats.keys()))
#     for i, y in enumerate(ys):
#         feats = y2feats[y]
#         mean_feats[i] = sum(feats) / len(feats)
#         ind2y[i] = y
#
#     return mean_feats.cpu(), ind2y

if __name__ == "__main__":
    target = sys.argv[1]
    model = models.resnet50(pretrained=False)
    # model.fc = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 128))
    checkpoint = torch.load(sys.argv[2], "cpu")
    model.fc = torch.nn.Identity()
    # model.fc = torch.nn.Linear(2048, checkpoint["state_dict"]["fc.weight"].shape[0])
    # state_dict = {k.replace("backbone.", ""):v for k, v in checkpoint["state_dict"].items()}
    log = model.load_state_dict(checkpoint["state_dict"], strict=False)
    print(log)
    # model.fc = SelfMap()
    model.to(device)

    model.eval()

    features = []
    file_index = 0

    transform = [transforms.Resize((128, 128)), transforms.ToTensor()]
    y2label = None
    if target == "oracle_token":
        items = get_oracle_tokens()
        target_paths = [item[0] for item in items]
        items = [(path, i) for i, path in enumerate(target_paths)]
    elif target.startswith('oracle'):
        target_paths = get_oracle_paths(f"./data/{target}")
    elif target == "predict":
        target_paths = get_standard_paths("./data/predict")
        items = [(path, i) for i, path in enumerate(target_paths)]
        transform = [transforms.Pad((16, 1), 255)] + transform
    elif target == "handa_8251":
        items, _, _ = get_handa_classes_data(contrast_ratios=list(range(0,100,10)), ratios=[1.0], get_specific_char="8251")
        target_paths = [item[0] for item in items]
        # transform = [transforms.RandomInvert(1.0)] + transform
    elif target == "handa_classes":
        # char = str(sys.argv[2])
        # target += "_" + char
        #TODO: try use correct items
        items, _, y2label = get_handa_classes_data(contrast_ratios=[0], data_root="../handa_classes_white", ratios=[0.8], get_specific_char=False, sort_imgs=True)
        preprocess = lambda x: x
        target_paths = [item[0] for item in items]
        # transform = [transforms.RandomInvert(1.0)] + transform
    elif target == "handa_all":
        items, _, _ = get_handa_classes_data(contrast_ratios=list(range(0, 100, 10)), data_root="./classes", ratios=[1.0], get_specific_char=False, use_all=True)
        items = [(items[i][0], i) for i in range(len(items))]
        target_paths = [item[0] for item in items]
    elif target == "handa_oracle_db":
        items, num_classes, _, path2id = get_oracle_db_data(ratios=[1.0], relabel=True, returnPath2Id=True)
        json.dump(path2id, open(f'./data/handa_oracle_db_{len(path2id)}_path2id.json', 'w'))
        target_paths = [item[0] for item in items]
        print(f"num_classes: {num_classes}")
    elif target == "jgwb":
        items, _, _ = get_jiaguwenbian_data(ratios=[1.0])
        target_paths = [item[0] for item in items]
        items = [(path, i) for i, path in enumerate(target_paths)]
    elif os.path.exists(target):
        target_paths = [path.strip() for path in open(target, 'r').readlines()]
        items = [(path, i) for i, path in enumerate(target_paths)]
    else:
        print("Please check args")
        exit(1)
    print(f"{target} has {len(target_paths)} images")
    if not os.path.exists('./feats'):
        os.mkdir("./feats")
    transform = transforms.Compose(transform)

    target = get_last_part(target)
    print(f"target: {target}")
    def getImageFun(path):
        img = Image.open(path).convert("L")
        return preprocess(img).convert("RGB")
    dataset = SupervisedOracleDataset(items, transform=transform, getImageFun=getImageFun)
    data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=256, shuffle=False,
            num_workers=16)
    # features = get_feats_from_paths(target_paths)
    features, y2ind = get_feats(model, data_loader, device)
    # if target == "handa_classes": # target_paths is rewritten to class folder
    #     target_paths = paths
    pos = 0
    while target_paths[0][pos] in [".", "/"]:
        pos += 1
    json.dump([path[pos:] for path in target_paths], open(f"feats/{target}_paths_{daystr()}.json", 'w'))
    torch.save((features, y2ind), f"feats/{target}_feats_{daystr()}.pth")

    # all_target.append(path)
# all_file = glob.glob(f"/home/chenqianyu/orcal/binary_train_dataset/*.png")
# all_target = []
# with torch.no_grad():
#     for path in tqdm(all_file):
#         name = path.split('/')[-1]
#         if name in target_names:
#             continue
#         img = get_img(path).to(device)
#         feat = model(img).squeeze(0)
#         features.append(feat)
#         all_target.append(path)

# features = torch.stack(features, 0)
# features = features / features.norm(dim=-1, keepdim=True)

#json.dump(all_atrget, open(f"/home/chenqianyu/orcal/feature/{file_index}/file_names.json", "w"))
# json.dump(target_paths, open(f"./all_file_names.json", "w"))
# print(len(all_target))