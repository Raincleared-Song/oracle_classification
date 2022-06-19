import argparse
import json

import numpy as np
import torch
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.nn import functional as F

from data.utils import get_handa_classes_data, get_oracle_db_data, get_supervised_data
from extract_features import get_feats
from data.oracle_dataset import SupervisedOracleDataset
from torchvision import transforms
from tqdm import tqdm

from utils import preprocess, get_last_part
from model import OCRModel
from PIL import Image

parser = argparse.ArgumentParser()
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--checkpoint', default="./checkpoints/handa_64_best.pth.tar")
parser.add_argument('--handa_root', default="/data/private/songchenyang/hanzi_filter/classes")
# parser.add_argument('--handa_feats', default="./feats/")
# parser.add_argument('--dual', default=False, action="store_true")
parser.add_argument("--use_best_feat", default=False, action="store_true")
parser.add_argument('--freq_limit', type=int, default=1e10)

def getImageFun(path):
    img = Image.open(path).convert("L")
    return preprocess(img).convert("RGB")
# getImageFun = None
def main():
    args = parser.parse_args()
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1
    device = args.device
    # model = models.resnet50(pretrained=False)
    # model.fc = torch.nn.Identity()
    # checkpoint = torch.load(args.checkpoint, "cpu")
    # state_dict = {k.replace("backbone.", ""):v for k, v in checkpoint["state_dict"].items()}
    # log = model.load_state_dict(state_dict, strict=False)
    # print(log)
    # model.to(device)
    # if args.dual:
    #     linear = torch.nn.Linear(2048, 3815)
    #     linear.weight = torch.nn.Parameter(state_dict["fc.weight"])
    #     linear.bias = torch.nn.Parameter(state_dict["fc.bias"])
    #     linear.to(device)
    ratios = [0.8,0.1,0.1]
    pred_imgs,_,te_imgs, num_classes, y2label = get_handa_classes_data(data_root=args.handa_root, ratios=ratios, relabel=True, sort_imgs=True)
    val_imgs, te_imgs, _, _ = get_oracle_db_data(ratios=[0.5, 0.5], relabel=False, y2label=y2label) # y2label 3815 没有复合字
    # y2imgs = dict()
    # for (img, y) in te_imgs:
    #     y = y2label[y]
    #     if y2imgs.get(y) is None:
    #         y2imgs[y] = []
    #     y2imgs[y].append(img)
    # with open("./log/y2imgs.json", 'w') as f:
    #     json.dump(y2imgs, f, indent=2)
    y2cnt = json.load(open(f"./data/handa_freq_{num_classes}.json"))
    print(f"te_imgs: {len(te_imgs)}")
    tmp = []
    print("fc:", args.freq_limit)
    for item in te_imgs:
        if y2cnt.get(str(item[1]),1e38) <= args.freq_limit:
            tmp.append(item)
    te_imgs = tmp
    print(f"te_imgs: {len(te_imgs)}")
    transform = [transforms.Resize((128, 128)), transforms.ToTensor()]
    # transform = [transforms.RandomInvert(1.0)] + transform

    # if "white" not in args.handa_root:
    #     transform = [transforms.RandomInvert(1.0)] + transform
    pred_dataloader = torch.utils.data.DataLoader(
            SupervisedOracleDataset(pred_imgs, transform=transforms.Compose(transform), getImageFun=getImageFun), batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    topk = (20,)
    model = OCRModel(checkpoint_path=args.checkpoint,
                     pred_dataloader=pred_dataloader,
                     device=device,
                     topk=topk,
                     use_best_feat=args.use_best_feat,
                     cache_path=f"{get_last_part(args.checkpoint)}@{get_last_part(args.handa_root)}_ratio={ratios[0]}.pkl",
                     y2label=y2label
                     )

    model.eval()
    # pred_feats, _ = get_feats(model, pred_dataloader)
    top1_test_accuracy = 0
    top5_test_accuracy = 0
    counter = 0
    for imgs in [val_imgs, te_imgs]:
        te_dataloader = torch.utils.data.DataLoader(
                SupervisedOracleDataset(imgs, transform=transforms.Compose(transform), getImageFun=getImageFun), batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
        for x_batch, y_batch in tqdm(te_dataloader):
            counter += int(y_batch.size(0))
            with torch.no_grad():
                pred = model(x_batch, y_batch)
                # feats = model(x_batch.to(device))
                # dists = euclidean_dist(feats, pred_feats).squeeze()
                # p_y = F.softmax(-dists, dim=1)
                # top1, top5, pred = accuracy(p_y, y_batch.to(device), topk=topk, return_pred=True)
                # if args.dual:
                #     #
                #     logits = F.softmax(linear(feats), dim=1)
                # for i, feat in enumerate(feats):
                #     label = int(y_batch[i])
            # top1_test_accuracy += top1
            # top5_test_accuracy += top5
        model.report()
    # top1_test_accuracy /= counter
    # top5_test_accuracy /= counter
    # print(counter)
    # print(
    #     f"Test Top{topk[0]} acc: {top1_test_accuracy}\tTop{topk[1]} acc: {top5_test_accuracy}")

if __name__ == "__main__":
    main()