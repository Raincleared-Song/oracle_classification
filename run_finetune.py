import argparse
import os

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import models
from data.oracle_dataset import SupervisedOracleDataset
from data.utils import get_supervised_data, get_oracle_labeled_data, get_handa_classes_data, get_handa_category_data, get_oracle_db_data, get_standard_paths, catGeneratedImage, get_jiaguwenbian_data
from utils import accuracy, save_checkpoint, get_last_part, Counter
from tqdm import tqdm
import torchvision.transforms as transforms

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='/home/zhangzeyuan/zeyuan/supervised_data/data1',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='oracle',
                    help='dataset name', choices=['stl10', 'cifar10', 'oracle'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n_views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu', default=0, type=int, help='Gpu index.')

# parser.add_argument('--batch_sizex', default=64, type=int, help='batch_size.')
parser.add_argument('--checkpoint', default="./runs/Nov24_14-36-53_node3/checkpoint_0200.pth.tar")

parser.add_argument('--num_classes', default=2387, type=int, help='number of classes.')
parser.add_argument('--data_root', default = "/data/private/songchenyang/hanzi_filter/classes")

parser.add_argument('--dataset', default="handa")
parser.add_argument('--ratios', default="0.8,0.1,0.1")
parser.add_argument('--combine', action="store_true", default=False)
parser.add_argument('--img_height', default=128)
parser.add_argument('--freq_limit', type=int, default=1e10)
parser.add_argument('--checkpoint_prefix', type=str, default="")
parser.add_argument('--second_root', default="")

def test(model, args, best_checkpoint, te_loader, y2label, test_data_name):
    if args.epochs > 0:
        checkpoint = torch.load(best_checkpoint)
        state_dict = {k.replace("backbone.", ""): v for k, v in checkpoint["state_dict"].items()}
        log = model.load_state_dict(state_dict, strict=False)
        print(log)
    model.eval()
    counter = 0
    log_path = f"./log/{test_data_name}_linear_{args.batch_size}.txt"
    f = open(log_path, 'w')
    f.write("Path\tlabel\tpreds\n")
    print(f"writing result to {log_path}")
    y2results = {}
    true_p_counter = Counter()
    false_p_counter = Counter()
    correct_cnt = dict()
    topk = [1,3,5,10]
    topk_cnters = [Counter() for _ in topk]
    for x_batch, y_batch, paths in te_loader:
        bz = int(y_batch.size(0))
        x_batch = x_batch.to(args.device)
        y_batch = y_batch.to(args.device)
        with torch.no_grad():
            logits = model(x_batch)
            p_y = F.softmax(logits, dim=1)
            res = accuracy(logits.detach(), y_batch.detach(), topk=topk, return_pred=True)
            vals, pred = res[:-1], res[-1]
            for i, val in enumerate(pred.T):
                pred_label = y2label[int(val[0])]
                pred_labels = [y2label[int(val[x])] for x in range(10)]
                true_label = y2label[int(y_batch[i])]
                # top_p_sum = float(torch.sum(p_y[i][val[:5]]))
                f.write(f"{paths[i]}    {true_label}    " + " ".join(pred_labels)+"\n")
                f.write(" ".join(map(lambda x: f"{x:.2f}", p_y[i][val[:10]].tolist())) + "\n")
                if true_label not in pred_labels:
                    # false_p_counter.add(top_p_sum)
                    pass
                else:
                    correct_cnt[true_label] = 1
                    # true_p_counter.add(top_p_sum)
                    y = int(y_batch[i])
                    if y2results.get(y) is None:
                        y2results[y] = []
                    y2results[y].append((paths[i], float(logits[i][0])))
            for i, val in enumerate(vals):
                topk_cnters[i].add(val, bz)
    #         top1_accuracy += top1
    #         top5_accuracy += top5
    # top1_accuracy /= counter
    # top5_accuracy /= counter
    prt_str = f"Test "
    for i, k in enumerate(topk):
        prt_str += f"Top{k} acc: {topk_cnters[i].average()}; "
    print(prt_str)
    # print(f"Test Top1 acc: {top1_accuracy}\tTop5 acc: {top5_accuracy}")
    print(f"correct classes: {len(correct_cnt)}")
    # f.write(f"{args.freq_limit}\t{top1_accuracy}\t{top5_accuracy}\n")
    f.close()
    # print(f"True p avg: {true_p_counter.average()}; False p avg: {false_p_counter.average()}")
    # for i in range(20):
    #     t = 0.8 + 0.01 * i
    #     print(f"Threshold {t}")
    #     print(f"False p threshold_cnt: {false_p_counter.threshold_cnt(t)}/{false_p_counter.cnt}")
    #     print(f"True p threshold_cnt: {true_p_counter.threshold_cnt(t)}/{true_p_counter.cnt}")
    # with open(f"./log/{test_data_name}_linear_protos", 'w') as f:
    #     for y in sorted(y2results.keys()):
    #         results = sorted(y2results[y], key=lambda x: x[1], reverse=True)
    #         f.write(results[0][0]+"\n")
    #     print("protos have been saved to:", f.name)

def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu}')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu = -1
    device = args.device
    ratios = list(map(float, args.ratios.split(",")))
    y2label = None
    img_height = args.img_height
    img_width = args.img_height if not args.combine else 2*args.img_height
    if args.dataset == "supervised":
        args.data_root = "data/supervised_data"
        tr_imgs, val_imgs, te_imgs, args.num_classes, y2label = get_supervised_data(data_paths=["data/supervised_data/data_1",
                                      # "data/supervised_data/data_2"
                                                                                 ],
                                      formats=["jpg", ], ratios=ratios)
        transform = transforms.Compose([transforms.ToTensor()])
    if args.dataset == "jgwb":
        tr_imgs, val_imgs, te_imgs, args.num_classes, y2label = get_jiaguwenbian_data(data_root=args.data_root,ratios=ratios,freq_limit=args.freq_limit)
        transform = transforms.Compose([transforms.Resize((img_width, img_height)), transforms.ToTensor()])
    elif args.dataset.startswith("oracle"):
        tr_imgs, val_imgs, te_imgs, args.num_classes, y2label = get_oracle_labeled_data(ratios=ratios)
        transform = transforms.Compose([transforms.Resize((img_width, img_height)), transforms.ToTensor()])
    elif args.dataset == "handa":
        tr_imgs, val_imgs, te_imgs, args.num_classes, y2label = get_handa_classes_data(ratios=ratios, data_root=args.data_root, freq_limit=args.freq_limit, sort_imgs=False)
        if args.second_root:
            tr_imgs2, val_imgs2, te_imgs2, _, _ = get_handa_classes_data(ratios=ratios,data_root=args.second_root,freq_limit=args.freq_limit)
            tr_imgs = tr_imgs + tr_imgs2
            val_imgs = val_imgs + val_imgs2
        if "white" in args.data_root:
            transform = transforms.Compose([transforms.Resize((img_width, img_height)), transforms.ToTensor()])
        else:
            transform=transforms.Compose([transforms.RandomInvert(1.0), transforms.Resize((img_width, img_height)), transforms.ToTensor()])
    elif args.dataset == "handa_category":
        tr_imgs, val_imgs, te_imgs, args.num_classes, y2label = get_handa_category_data(ratios=ratios, data_root=args.data_root)
        transform=transforms.Compose([transforms.RandomInvert(1.0), transforms.Resize((img_width, img_height)), transforms.ToTensor()])
    elif args.dataset == "handa_oracle_db":
        tr_imgs, val_imgs, te_imgs, args.num_classes, y2label = get_oracle_db_data(ratios=ratios,
                                                                                       freq_limit=args.freq_limit)
        transform = transforms.Compose(
            [transforms.Resize((img_width, img_height)), transforms.ToTensor()])
    elif args.dataset == "predict":
        imgs = get_standard_paths(args.data_root)
        tr_imgs = []
        val_imgs = []
        args.num_classes = len(imgs)
        y2label = {i: get_last_part(imgs[i]) for i in range(args.num_classes)}
        te_imgs = [(img, i) for i, img in enumerate(imgs)]
        transform = transforms.Compose([transforms.Resize((img_width, img_height)), transforms.ToTensor()])
    getImageFun = catGeneratedImage if args.combine else None
    tr_dataset = SupervisedOracleDataset(tr_imgs, transform=transform, getImageFun=getImageFun)
    val_dataset = SupervisedOracleDataset(val_imgs, transform=transform, getImageFun=getImageFun)
    te_dataset = SupervisedOracleDataset(te_imgs, transform=transform, return_path=True, getImageFun=getImageFun)

    print(f"train num: {len(tr_dataset)}")
    print(f"valid num: {len(val_dataset)}")
    print(f"test num: {len(te_dataset)}")
    print(f"num_classes: {args.num_classes}")
    print(f"batch_size: {args.batch_size}")
    print(f"freq_limit:{args.freq_limit}")

    if len(tr_dataset) > 0:
        train_loader = torch.utils.data.DataLoader(
        tr_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    if len(val_dataset) > 0:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    if len(te_dataset) > 0:
        te_loader = torch.utils.data.DataLoader(
        te_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.second_root:
        te_dataset2 = SupervisedOracleDataset(te_imgs2, transform=transform, return_path=True, getImageFun=getImageFun)
        te_loader2 = torch.utils.data.DataLoader(
            te_dataset2, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)


    if os.path.exists(args.checkpoint):
        model = models.resnet50(pretrained=False, num_classes=args.num_classes)
        # model.fc = torch.nn.Linear(2048, args.num_classes)
        checkpoint = torch.load(args.checkpoint, "cpu")
        state_dict = {k.replace("backbone.", ""): v for k, v in checkpoint["state_dict"].items()}
        # torch.set_printoptions(profile="full")
        # for k, v in state_dict.items():
        #     if "weight" in k and "conv" in k:
        #         print(k, v)
        # exit(0)
        #     state_dict[k] += torch.rand_like(state_dict[k])
        log = model.load_state_dict(state_dict, strict=False)
        # print(model.fc.weight)
        print(log)
    else:
        print("checkpoint not found!")
        model = models.resnet50(pretrained=False, num_classes=args.num_classes)
        # print(model.state_dict())
        # model.fc = torch.nn.Linear(2048, args.num_classes)
        # exit(1)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    # optimizer.load_state_dict(checkpoint["optimizer"])
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # print(len(train_loader))
    max_val_acc = 0
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    best_checkpoint = os.path.join('./checkpoints', f'{args.checkpoint_prefix}{args.dataset}_{args.batch_size}_best.pth.tar')
    topk = [1,5]
    for epoch in range(args.epochs):
        topk_cnters = [Counter() for _ in topk]
        model.train()
        # counter = 0
        for x_batch, y_batch in tqdm(train_loader):
            bz = int(y_batch.size(0))
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            vals = accuracy(logits.detach(), y_batch.detach(), topk=topk)
            for i, val in enumerate(vals):
                topk_cnters[i].add(val, bz)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print_str = f"Epoch {epoch} train "
        for i, k in enumerate(topk):
            print_str += f"Top{k} acc: {topk_cnters[i].average()}; "
        print(print_str)
        topk_cnters = [Counter() for _ in topk]
        model.eval()
        counter = 0
        for x_batch, y_batch in val_loader:
            bz = int(y_batch.size(0))
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            with torch.no_grad():
                logits = model(x_batch)
                vals = accuracy(logits.detach(), y_batch.detach(), topk=topk)
                for i, val in enumerate(vals):
                    topk_cnters[i].add(val, bz)
        # top1_val_accuracy /= counter
        # top5_val_accuracy /= counter
        top1_val_accuracy = topk_cnters[0].average()
        max_val_acc = max(max_val_acc, top1_val_accuracy)
        print_str = f"Epoch {epoch} valid "
        for i, k in enumerate(topk):
            print_str += f"Top{k} acc: {topk_cnters[i].average()}; "
        print(print_str)
        # print(f"\t valid Top1 acc: {top1_val_accuracy}\tTop5 acc: {top5_val_accuracy}")

        epochCounter = epoch + 1
        is_best_checkpoint = max_val_acc<=top1_val_accuracy
        if (epochCounter // 10)*10 == epochCounter or is_best_checkpoint:
            # save model checkpoints
            checkpoint_name = f'{args.checkpoint_prefix}{args.dataset}_{args.batch_size}_{epochCounter:04d}.pth.tar'
            save_checkpoint({
                'args': args,
                'epoch': epochCounter,
                'arch': args.arch,
                'acc': top1_val_accuracy,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            },
                is_best=is_best_checkpoint,
                filename=os.path.join('./checkpoints', checkpoint_name),
                model_best=best_checkpoint)
    # test
    print("Running test")
    test(model, args, best_checkpoint, te_loader, y2label, get_last_part(args.data_root))
    if args.second_root:
        test(model, args, best_checkpoint, te_loader2, y2label, get_last_part(args.second_root))

if __name__ == "__main__":
    main()
