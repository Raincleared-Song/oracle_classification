import argparse
from datetime import datetime
import os

import torch
import torch.backends.cudnn as cudnn
from torchvision import models, transforms
from torch.nn import functional as F
from data.oracle_dataset import SupervisedOracleDataset
from data.utils import get_supervised_data, get_handa_classes_data, get_oracle_labeled_data, get_standard_paths, get_jiaguwenbian_data
from utils import accuracy, euclidean_dist, get_last_part, Counter, get_feats
from tqdm import tqdm
import logging

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
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--checkpoint', default="proto_checkpoints/handa_50.pth.tar")
parser.add_argument('--dataset', default="handa")
parser.add_argument('--data_root', default = "/data/private/songchenyang/hanzi_filter/classes")
parser.add_argument('--gpu', default=0, type=int, help='Gpu index.')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--dim_feature', default=2048)
parser.add_argument('--freq_limit', type=int, default=1e10)
parser.add_argument('--test_all', default=False, action="store_true")
parser.add_argument('--ratios', default="0.8,0.1,0.1")
parser.add_argument('--dual', default=False, action="store_true")

def test(dataloader, model, y2label, args, prototypes, y2ind, linear=None):
    # ind2label = list(label2ind.keys())
    model.eval()
    f = open("./log/proto_test_result.txt", 'a')
    y2results = {}
    ind2y = {v:k for k, v in y2ind.items()}
    correct_cnt = dict()
    with torch.no_grad():
        top1_test_accuracy = 0
        top5_test_accuracy = 0
        counter = 0
        # true_p_counter = Counter()
        # false_p_counter = Counter()
        for x, y_batch, paths in tqdm(dataloader):
            counter += int(y_batch.size(0))
            y_batch = y_batch.to(args.device)
            y2 = torch.tensor([y2ind[int(label)] for label in y_batch], device=args.device)
            querys = model(x.to(args.device))
            dists = euclidean_dist(querys, prototypes)
            p_y = F.softmax(-dists, dim=1)
            top1, top5, pred = accuracy(p_y, y2, topk=(1, 5), return_pred=True)
            # print(pred.T)
            # exit(0)
            for i, topk in enumerate(pred.T):
                # pred_label = y2label[ind2label[int(topk[0])]]
                pred_labels = [y2label[ind2y[int(topk[x])]] for x in range(5)]
                true_label = y2label[int(y_batch[i])]
                # top_p_sum = float(torch.sum(p_y[i][topk[:5]]))
                pred_label = pred_labels[0]
                # f.write(f"{paths[i]} {pred_label} {true_label}\n")
                if (true_label not in pred_labels):
                    # false_p_counter.add(top_p_sum)
                    pass
                else:
                    correct_cnt[true_label] = 1
                    # true_p_counter.add(top_p_sum)
                    y = int(y_batch[i])
                    score = float(torch.max(p_y[i]))
                    if y2results.get(y, (0,0))[1] < score:
                        y2results[y] = (paths[i], score, querys[i])
            top1_test_accuracy += top1
            top5_test_accuracy += top5
            # if linear:
            #     logits = F.softmax(linear(querys), dim=1)
            #     _, _, pred = accuracy(logits.detach(), y_batch.detach(), topk=(1, 5), return_pred=True)
            #     for i, topk in enumerate(pred.T):
            #         pred_label = y2label[int(topk[0])]
            #         true_label = y2label[int(y_batch[i])]
            #         if pred_label == true_label:
            #             y = int(y_batch[i])
            #             score = float(torch.max(logits[i]))
            #             if y2results.get(y, (0, 0))[1] < score:
            #                 y2results[y] = (paths[i], score, querys[i])
        top1_test_accuracy /= counter
        top5_test_accuracy /= counter
        print(counter)
        print(f"Test Top1 acc: {top1_test_accuracy}\tTop5 acc: {top5_test_accuracy}")
        # print(f"True p avg: {true_p_counter.average()}; False p avg: {false_p_counter.average()}")
        print(f"correct classes: {len(correct_cnt)}")
        # f.write(f"{args.freq_limit}\t{top1_test_accuracy}\t{top5_test_accuracy}\n")
        # for i in range(20):
        #     t = 0.8 + 0.01 * i
        #     print(f"Threshold {t}")
        #     print(f"False p threshold_cnt: {false_p_counter.threshold_cnt(t)}/{false_p_counter.cnt}")
        #     print(f"True p threshold_cnt: {true_p_counter.threshold_cnt(t)}/{true_p_counter.cnt}")
        f.close()
        # model_type = "dual" if args.dual else "ProtoNet"
        # feats = []
        # target_paths = []
        # with open(f"./log/{get_last_part(args.data_root)}_{model_type}_protos", 'w') as f:
        #     for y in sorted(y2results.keys()):
        #         # results = sorted(y2results[y], key=lambda x: x[1], reverse=True)
        #         f.write(y2results[y][0] + "\n") #note: use the best result as prototype
        #         feats.append(y2results[y][2])
        #         target_paths.append(y2results[y][0])
        #     print("protos have been saved to:", f.name)
        # torch.save(torch.stack(feats).cpu(), f"feats/{get_last_part(args.data_root)}_{model_type}_feats.pth")
        # ind = 0
        # while target_paths[0][ind] in [".", "/"]:
        #     ind += 1
        # json.dump([path[ind:] for path in target_paths], open(f"feats/{get_last_part(args.data_root)}_{model_type}_paths.json", 'w'))


def main():
    args = parser.parse_args()
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu}')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1
    device = args.device
    ratios = list(map(float, args.ratios.split(",")))
    if not os.path.exists("log"):
        os.mkdir('log')
    logging.basicConfig(filename=f"./log/{datetime.now().strftime('%Y%m%d%H%M%S.log')}", level=logging.INFO)

    transform = transforms.Compose([transforms.ToTensor()])
    if args.dataset == "handa":
        tr_imgs, val_imgs, te_imgs, args.num_classes, y2label = get_handa_classes_data(ratios=ratios, data_root=args.data_root, freq_limit=args.freq_limit, sort_imgs=False)
        if "white" in args.data_root:
            transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        else:
            transform=transforms.Compose([transforms.RandomInvert(1.0), transforms.Resize((128, 128)), transforms.ToTensor()])
    elif args.dataset == "oracle":
        tr_imgs, val_imgs, te_imgs, args.num_classes, y2label = get_oracle_labeled_data(ratios=ratios)
        transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    elif args.dataset == "jgwb":
        tr_imgs, val_imgs, te_imgs, args.num_classes, y2label = get_jiaguwenbian_data(ratios=ratios)
        transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    elif args.dataset == 'supervised':
        # train on supervised data
        tr_imgs, val_imgs, te_imgs, args.num_classes, y2label = get_supervised_data(data_paths=["data/supervised_data/data_1",
                                          "data/supervised_data/data_2"],
                                          formats=["jpg", "png"], ratios=ratios)
    elif args.dataset == "predict":
        imgs = get_standard_paths(args.data_root)
        tr_imgs = []
        val_imgs = []
        args.num_classes = len(imgs)
        y2label = {i: get_last_part(imgs[i]) for i in range(args.num_classes)}
        te_imgs = [(img, i) for i, img in enumerate(imgs)]
        transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

    else:
        print(f"Unkown dataset: {args.dataset}")
        exit(1)

    tr_dataset = SupervisedOracleDataset(tr_imgs, transform=transform)
    if args.test_all:
        te_imgs = tr_imgs + val_imgs + te_imgs
    te_dataset = SupervisedOracleDataset(te_imgs, transform=transform, return_path=True)
    print(f"te_dataset len: {len(te_dataset)}")
    tr_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=args.batch_size, num_workers=args.workers)
    te_dataloader = torch.utils.data.DataLoader(te_dataset, batch_size=args.batch_size, num_workers=args.workers)
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Identity()
    # model.fc = torch.nn.Linear(2048, args.dim_feature)
    print(f"Loading {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, "cpu")
    state_dict = {k.replace("backbone.", ""): v for k, v in checkpoint["state_dict"].items()}
    log = model.load_state_dict(state_dict, strict=False)
    model.to(device)
    print(log)
    # if args.dual:
    #     linear = torch.nn.Linear(2048, args.num_classes)
    #     linear.weight = torch.nn.Parameter(state_dict["fc.weight"])
    #     linear.bias = torch.nn.Parameter(state_dict["fc.bias"])
    #     linear.to(device)
    # else:
    #     linear = None
    prototypes, y2ind = get_feats(model, tr_dataloader, device,
                                  cache_path=f"{get_last_part(args.checkpoint)}@{get_last_part(args.data_root)}_ratio={ratios[0]}")
    test(te_dataloader, model, y2label, args, prototypes, y2ind)

if __name__ == "__main__":
    main()
