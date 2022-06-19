import argparse
import json
import os
import numpy as np
import torch
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.nn import functional as F

from data.utils import get_handa_classes_data, get_oracle_db_data
from extract_features import get_feats
from data.oracle_dataset import SupervisedOracleDataset
from torchvision import transforms
from tqdm import tqdm

from utils import preprocess, get_last_part, accuracy, save_checkpoint, Counter
from model import DualModel
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
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--checkpoint', default="./checkpoints/trans_mix_handa_256_best.pth.tar")
parser.add_argument('--handa_root', default="./classes")
# parser.add_argument('--handa_feats', default="./feats/")
# parser.add_argument('--dual', default=False, action="store_true")
parser.add_argument("--use_best_feat", default=False, action="store_true")
parser.add_argument('--freq_limit', type=int, default=1e10)
parser.add_argument('--gpu', default=0, type=int, help='Gpu index.')

def getImageFun(path):
    img = Image.open(path).convert("L")
    return preprocess(img).convert("RGB")
getImageFun = None
def main():
    args = parser.parse_args()
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu}')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu = -1
    device = args.device
    ratios = [0.8,0.1,0.1]
    pred_imgs, tr_imgs, te_imgs, num_classes, y2label = get_handa_classes_data(data_root=args.handa_root, ratios=ratios, relabel=True, sort_imgs=True)
    # te_imgs, _, _ = get_oracle_db_data(ratios=[1.0], relabel=False, y2label=y2label) # y2label 3815 没有复合字
    y2cnt = json.load(open(f"./data/handa_freq_{num_classes}.json"))
    print(f"te_imgs: {len(te_imgs)}")
    tmp = []
    for item in te_imgs:
        if y2cnt.get(str(item[1]),1e38) <= args.freq_limit:
            tmp.append(item)
    te_imgs = tmp
    print(f"te_imgs: {len(te_imgs)}")
    transform = [transforms.RandomInvert(1.0), transforms.Resize((128, 128)), transforms.ToTensor()]
    tr_dataloader = torch.utils.data.DataLoader(
            SupervisedOracleDataset(tr_imgs, transform=transforms.Compose(transform), getImageFun=getImageFun), batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    te_dataloader = torch.utils.data.DataLoader(
            SupervisedOracleDataset(te_imgs, transform=transforms.Compose(transform), getImageFun=getImageFun), batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    # if "white" not in args.handa_root:
    #     transform = [transforms.RandomInvert(1.0)] + transform
    pred_dataloader = torch.utils.data.DataLoader(
            SupervisedOracleDataset(pred_imgs, transform=transforms.Compose(transform), getImageFun=getImageFun), batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    topk = (1, 5)
    model = DualModel(checkpoint_path=args.checkpoint,
                      pred_dataloader=pred_dataloader,
                      cache_path=f"{get_last_part(args.checkpoint)}@{get_last_part(args.handa_root)}_ratio={ratios[0]}.pkl",
                      device=args.device)
    model.to(device)
    # train
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    max_val_acc = 0
    # test
    model.eval()
    top1_val_acc = 0
    top5_val_acc = 0
    counter = 0
    for x_batch, y_batch in tqdm(te_dataloader):
        counter += int(y_batch.size(0))
        with torch.no_grad():
            p_y = model(x_batch.to(device))
            top1, top5, pred = accuracy(p_y, y_batch.to(device), topk=topk, return_pred=True)
            # for i, feat in enumerate(feats):
            #     label = int(y_batch[i])
        top1_val_acc += top1
        top5_val_acc += top5
    top1_val_acc /= counter
    max_val_acc = max(max_val_acc, top1_val_acc)
    top5_val_acc /= counter
    print(counter)
    print(
        f"Test Top{topk[0]} acc: {top1_val_acc}\tTop{topk[1]} acc: {top5_val_acc}")

    for epoch in range(args.epochs):
        top1_train_accuracy = 0
        top5_train_accuracy = 0
        model.train()
        counter = 0
        train_loss = Counter()
        for x_batch, y_batch in tqdm(tr_dataloader):
            counter += int(y_batch.size(0))
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            train_loss.add(float(loss.detach()))
            top1, top5 = accuracy(logits.detach(), y_batch.detach(), topk=(1, 5))
            top1_train_accuracy += top1
            top5_train_accuracy += top5
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        top1_train_accuracy /= counter
        top5_train_accuracy /= counter
        print(f"Epoch {epoch} train Top1 acc: {top1_train_accuracy}\tTop5 acc: {top5_train_accuracy}, loss: {train_loss.average()}")

        # test
        model.eval()
        top1_val_acc = 0
        top5_val_acc = 0
        counter = 0
        valid_loss = Counter()
        for x_batch, y_batch in tqdm(te_dataloader):
            counter += int(y_batch.size(0))
            with torch.no_grad():
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                p_y = model(x_batch)
                valid_loss.add(criterion(p_y, y_batch))
                top1, top5, pred = accuracy(p_y, y_batch.to(device), topk=topk, return_pred=True)
                # for i, feat in enumerate(feats):
                #     label = int(y_batch[i])
            top1_val_acc += top1
            top5_val_acc += top5
        top1_val_acc /= counter
        max_val_acc = max(max_val_acc, top1_val_acc)
        top5_val_acc /= counter
        print(counter)
        print(
            f"Valid Top{topk[0]} acc: {top1_val_acc}\tTop{topk[1]} acc: {top5_val_acc}, loss: {valid_loss.average()}")

        epochCounter = epoch + 1
        is_best_checkpoint = max_val_acc<=top1_val_acc
        if (epochCounter // 10)*10 == epochCounter or is_best_checkpoint:
            # save model checkpoints
            checkpoint_name = os.path.join('./checkpoints', f'dual_model_best.pth.tar')
            save_checkpoint({
                'args': args,
                'epoch': epochCounter,
                'arch': args.arch,
                'acc': top1_val_acc,
                'state_dict': model.trainable_parameters(),
                'optimizer': optimizer.state_dict(),
            },
                filename=checkpoint_name
                )

if __name__ == "__main__":
    main()