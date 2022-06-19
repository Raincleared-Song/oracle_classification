import argparse
import os
from PIL import Image
import json
from tqdm import tqdm
import torch
from torchvision import models
import torch.backends.cudnn as cudnn

from data.oracle_dataset import SupervisedOracleDataset
from data.utils import get_oracle_labeled_data
import torchvision.transforms as transforms
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--gpu', default=0, type=int, help='Gpu index.')

parser.add_argument('--dataset', default="supervised")
parser.add_argument('--ratios', default="0.8,0.1,0.1")
parser.add_argument('--checkpoint')

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
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats

def long_img(images, dest_path, s=200):
    # print(images)

    # 单幅图像尺寸
    width, height = images[0].size
    width, height = images[0].resize((s, s)).size

    # 创建空白长图
    result = Image.new(images[0].mode, (width*len(images), height), color=0)    # 默认填充一张图片的像素值是黑色
    # print(type(result), result.size)   # <class 'PIL.Image.Image'> (500, 3000)

    # 拼接
    for i, img in enumerate(images):
        img = img.resize((s, s))
        result.paste(img, box=(i*width, 0))   # 把每一张img粘贴到空白的图中，注意，如果图片的宽度大于空白图的长度

    result.save(dest_path)


if __name__ == "__main__":
    # target_paths = json.load(open("./feats/target_paths.json"))  # 14752
    # result = json.load(open("./result.json"))
    if not os.path.exists('./images'):
        os.mkdir('images')
    args = parser.parse_args()
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1
    device = args.device
    tr_imgs, val_imgs, te_imgs, args.num_classes, y2char = get_oracle_labeled_data(ratios=(0.8,0.1,0.1))
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    te_dataset = SupervisedOracleDataset(te_imgs, return_path=True)

    te_loader = torch.utils.data.DataLoader(
        te_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(2048, args.num_classes)
    model.to(device)
    checkpoint = torch.load(os.path.join('./checkpoints', args.checkpoint))
    state_dict = {k.replace("backbone.", ""): v for k, v in checkpoint["state_dict"].items()}
    log = model.load_state_dict(state_dict, strict=False)
    print(log)
    model.eval()
    top1_accuracy, top5_accuracy = 0, 0
    counter = 0
    for x, y, paths in te_loader:
        counter += 1
        x = x[0]
        with torch.no_grad():
            logits = model(x.to(device))
            _, pred = logits.topk(100, 1, True, True)
        images = [transforms.functional.to_pil_image(x)]
        for y in pred.squeeze().tolist():
            unicode = f"{ord(y2char[y]):X}"
            image = Image.open(f'/home/zhangzeyuan/zeyuan/SimCLR-release/predict/u{unicode}.png')
            images.append(image)
        long_img(images, f"./images/oracle_{x}_match.png")
        if counter >= 100:
            break

    # for target_path in tqdm(target_paths[:100]):
    #     paths = [target_path]
    #     paths += result[target_path][:10]
    #     name = target_path.split("/")[-1].split(".")[0]
    #     images = []
    #     for img in paths:
    #         image = Image.open(img)
    #         images.append(image)
    #     long_img(images, f"./images/{name}_match.png")
    #
