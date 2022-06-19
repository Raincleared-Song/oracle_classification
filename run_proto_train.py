import argparse
import os

import torch
import torch.backends.cudnn as cudnn
from torchvision import models, transforms
from data.oracle_dataset import SupervisedOracleDataset
from data.prototypical_batch_sampler import PrototypicalBatchSampler
from data.utils import get_supervised_data, get_handa_classes_data, get_oracle_labeled_data, get_jiaguwenbian_data
from utils import accuracy, save_checkpoint, euclidean_dist
from tqdm import tqdm
import numpy as np
from torch.nn import functional as F
from math import ceil

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
parser.add_argument('--wd', '--weight-decay', default=0.0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-nsTr', '--num_support_tr',
                    type=int,
                    help='number of samples per class to use as support for training, default=5',
                    default=1)
parser.add_argument('-nqTr', '--num_query_tr',
                    type=int,
                    help='number of samples per class to use as query for training, default=5',
                    default=1)
parser.add_argument('-cTr', '--classes_per_it_tr',
                    type=int,
                    help='number of random classes per episode for training, default=20',
                    default=100)
parser.add_argument('-its', '--iterations',
                    type=int,
                    help='number of episodes per epoch, default=100',
                    default=100)
parser.add_argument('--datasets', default='supervised,handa')
parser.add_argument('--checkpoint', default="./runs/Nov24_14-36-53_node3/checkpoint_0200.pth.tar")
parser.add_argument('--checkpoint_prefix', default='')
parser.add_argument('--lr_scheduler_gamma', default=1.0)
parser.add_argument('--lr_scheduler_step', default=20)
parser.add_argument('--data_root', default = "/data/private/songchenyang/hanzi_filter/classes")
parser.add_argument('--gpu', default=0, type=int, help='Gpu index.')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--dim_feature', default=64)
parser.add_argument('--grad_upd_iter', default=1, type=int)

def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)

def init_dataloader(opt, dataset, mode):
    sampler = init_sampler(opt, dataset.y, mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=opt.workers)
    return dataloader

def init_sampler(opt, labels, mode):
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    num_support=opt.num_support_tr,
                                    iterations=opt.iterations)
def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

def find_support_indices(y_batch):
    indices = [0]
    found_last = False
    for i in range(len(y_batch)-1):
        if y_batch[i] < y_batch[i+1]:
            indices.append(i + 1)
        if y_batch[i] > y_batch[i+1]:
            indices.append(i + 1)
            found_last = True
            break
    if not found_last:
        indices.append(len(y_batch))
    return indices

def sum_by_indices(input: torch.Tensor, indices:list):
    res = []
    for i in range(len(indices)-1):
        res.append(input[indices[i]:indices[i+1]].mean(0))
    return torch.stack(res)

def validate(args, val_loader, model, device):
    top1_val_accuracy = 0
    top5_val_accuracy = 0
    model.eval()
    counter = 0
    no_query_cnt = 0
    for x_batch, y_batch in tqdm(val_loader):
        if args.classes_per_it_tr == len(y_batch):
            no_query_cnt += 1
            continue
        support_indices = find_support_indices(y_batch)
        support_num = support_indices[-1]
        if support_num == len(y_batch):
            continue
        counter += int(y_batch.size(0)-support_num)
        y2ind = {int(y_batch[support_indices[i]]): i for i in range(len(support_indices)-1)}
        x = x_batch.to(device)
        y = y_batch[support_num:]
        y = torch.tensor(list(map(lambda l: y2ind[l], y.tolist())), device=device)
        with torch.no_grad():
            logits = model(x)
            prototypes = sum_by_indices(logits[:support_num], support_indices)  # N
            dists = euclidean_dist(logits[support_num:], prototypes)  # MxN
            log_p_y = F.log_softmax(-dists, dim=1)  # MxN

            # y_protos, y = torch.unique(y, sorted=True,
            #                            return_inverse=True)  # set y as the position of the corresponding prototype
            top1, top5 = accuracy(log_p_y.detach(), y, topk=(1, 5))
            top1_val_accuracy += top1
            top5_val_accuracy += top5
    top1_val_accuracy /= counter
    top5_val_accuracy /= counter
    if no_query_cnt > 0:
        print("no_query_cnt:", no_query_cnt)
    print(f"\t valid Top1 acc: {top1_val_accuracy}\tTop5 acc: {top5_val_accuracy}")
    return top1_val_accuracy

def train(args, tr_loader, val_loader, model, device, used_datasets):
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.NLLLoss().to(device)
    lr_scheduler = init_lr_scheduler(args, optimizer)
    max_val_acc = 0
    bs = args.batch_size
    if "checkpoints" in args.checkpoint:
        max_val_acc = validate(args, val_loader, model, device)
    for epoch in range(1, args.epochs+1):
        model.train()
        top1_train_accuracy = 0
        top5_train_accuracy = 0
        counter = 0
        optimizer.zero_grad()
        for iter, batch in enumerate(tqdm(tr_loader)):
            x_batch, y_batch = batch
            num = int(x_batch.size(0))
            counter += num
            support_indices = find_support_indices(y_batch)
            support_num = support_indices[-1]
            y2ind = {int(y_batch[support_indices[i]]): i for i in range(len(support_indices) - 1)}
            x = x_batch.to(device)
            y = torch.tensor(list(map(lambda l: y2ind[l], y_batch.tolist())), device=device)
            # steps = ceil(num / bs)
            # log_p_y = []
            # for i in range(steps):
            #     log_p_y.append(model(x[i*bs: min((i+1)*bs, num)]))
            # log_p_y = torch.cat(log_p_y, dim=0)
            logits = model(x)
            prototypes = sum_by_indices(logits[:support_num], support_indices)  # N
            # our version uses all tensors as query
            dists = euclidean_dist(logits, prototypes)  # MxN
            log_p_y = F.log_softmax(-dists, dim=1)  # MxN
            # During sampling, y_protos (at the beginning of y) are already sorted by classes
            # y_protos, y = torch.unique(y, sorted=True, return_inverse=True) # set y as the position of the corresponding prototype
            loss = criterion(log_p_y, y)
            top1, top5 = accuracy(log_p_y.detach(), y.detach(), topk=(1, 5))
            top1_train_accuracy += top1
            top5_train_accuracy += top5
            loss /= args.grad_upd_iter
            loss.backward()
            if (iter+1) % args.grad_upd_iter == 0:
                optimizer.step()
                optimizer.zero_grad()
        top1_train_accuracy /= counter
        top5_train_accuracy /= counter
        print(
            f"Epoch {epoch}\tTop1 acc: {top1_train_accuracy}\tTop5 acc: {top5_train_accuracy}")
        top1_val_accuracy = validate(args, val_loader, model, device)
        max_val_acc = max(max_val_acc, top1_val_accuracy)

        # save model checkpoints
        if (epoch // 10) * 10 == epoch or max_val_acc<=top1_val_accuracy:
            checkpoint_prefix = args.checkpoint_prefix + '+'.join(used_datasets)
            checkpoint_name = checkpoint_prefix + f'_{epoch:04d}.pth.tar'
            save_checkpoint({
                'args': args,
                'epoch': epoch,
                'arch': args.arch,
                'acc': top1_train_accuracy,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            },
                is_best=max_val_acc<=top1_val_accuracy,
                filename=os.path.join('./proto_checkpoints', checkpoint_name),
                model_best=f'proto_checkpoints/{checkpoint_prefix}_best.pth.tar')
    lr_scheduler.step()


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

    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()
    # model.fc = torch.nn.Linear(2048, args.dim_feature)
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, "cpu")
        state_dict = {k.replace("backbone.", ""): v for k, v in checkpoint["state_dict"].items()}
        log = model.load_state_dict(state_dict, strict=False)
        print(log)
    model.to(device)
    if not os.path.exists('proto_checkpoints'):
        os.mkdir('proto_checkpoints')
    used_datasets=[]
    if 'supervised' in args.datasets:
        # train on supervised data
        tr_imgs, val_imgs, num_classes, _ = get_supervised_data(data_paths=["data/supervised_data/data_1",
                                          # "data/supervised_data/data_2"
                                                                            ],
                                          formats=["jpg", "png"], ratios=(0.9,0.1))
        tr_dataset = SupervisedOracleDataset(tr_imgs)
        val_dataset = SupervisedOracleDataset(val_imgs)
        tr_dataloader = init_dataloader(args, tr_dataset, 'train')
        val_dataloader = init_dataloader(args, val_dataset, 'train')
        used_datasets.append('supervised')
        train(args, tr_dataloader, val_dataloader, model, device, used_datasets)
    if "jgwb" in args.datasets:
        tr_imgs, val_imgs, num_classes, y2label = get_jiaguwenbian_data(ratios=(0.8, 0.1))
        transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        tr_dataset = SupervisedOracleDataset(tr_imgs, transform=transform)
        val_dataset = SupervisedOracleDataset(val_imgs, transform=transform)
        tr_dataloader = init_dataloader(args, tr_dataset, 'train')
        val_dataloader = init_dataloader(args, val_dataset, 'train')
        print(len(val_dataloader))
        used_datasets.append('jgwb')
        train(args, tr_dataloader, val_dataloader, model, device, used_datasets)

    if 'handa' in args.datasets:
        # train on handa data`
        tr_imgs, val_imgs, num_classes, y2char = get_handa_classes_data(data_root=args.data_root, ratios=(0.8,0.1))
        tr_dataset = SupervisedOracleDataset(tr_imgs, transform=transforms.Compose([transforms.RandomInvert(1.0), transforms.Resize((128,128)), transforms.ToTensor()]))
        val_dataset = SupervisedOracleDataset(val_imgs, transform=transforms.Compose([transforms.RandomInvert(1.0), transforms.Resize((128,128)), transforms.ToTensor()]))
        tr_dataloader = init_dataloader(args, tr_dataset, 'train')
        val_dataloader = init_dataloader(args, val_dataset, 'train')
        used_datasets.append('handa')
        train(args, tr_dataloader, val_dataloader, model, device, used_datasets)
    if 'oracle' in args.datasets:
        tr_imgs, val_imgs, num_classes, _ = get_oracle_labeled_data()
        #TODO: try padding instead of resize
        tr_dataset = SupervisedOracleDataset(tr_imgs, transform=transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()]))
        val_dataset = SupervisedOracleDataset(val_imgs, transform=transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()]))
        tr_dataloader = init_dataloader(args, tr_dataset, 'train')
        val_dataloader = init_dataloader(args, val_dataset, 'train')
        used_datasets.append('oracle')
        train(args, tr_dataloader, val_dataloader, model, device, used_datasets)

if __name__ == "__main__":
    main()