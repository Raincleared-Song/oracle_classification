import json
import os.path
import sys
import torch
from tqdm import tqdm
from utils import device
from PIL import Image
from torchvision import transforms
from utils import euclidean_dist



def get_img(path):
    transform = transforms.ToTensor()
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)

def get_feats_from_paths(paths, model):
    feats = []
    with torch.no_grad():
        for path in tqdm(paths):
            img = get_img(path).to(device)
            feat = model(img).squeeze(0)
            feats.append(feat)
    feats = torch.stack(feats, 0)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats

target = sys.argv[1]
predict = sys.argv[2]
print(f"Openning features of {target}")
target_paths = json.load(open(f"./feats/{target}_paths.json")) # 14752
pred_paths = json.load(open(f"./feats/{predict}_paths.json")) # 6203
print("target_len:", len(target_paths))
print("pred_len:", len(pred_paths))
result = {}
result2 = {}
target_feats = torch.load(f"./feats/{target}_feats.pth", map_location=device)
pred_feats = torch.load(f"./feats/{predict}_feats.pth", map_location=device)
path2dist = {}
# pred_feats = pred_feats.T
for idx, path in enumerate(tqdm(target_paths)):
    # sim = torch.matmul(target_feats[idx], pred_feats).squeeze() #6203
    dists = euclidean_dist(target_feats[idx].unsqueeze(0), pred_feats).squeeze() #6203
    _, indices = torch.sort(dists)
    result[path] = [pred_paths[i] for i in indices.tolist()[:100]] # path with prefix
    path2dist[path] = float(dists[0])
    # result2[path] = [pred_paths[i].split("/")[-1] for i in indices.tolist()[:100]] # path without prefix

if not os.path.exists("result"):
    os.mkdir("result")
result = {k: v for k,v in sorted(result.items(), key=lambda x: path2dist[x[0]])}
json.dump(result, open(f"./result/{target}_{predict}.json", 'w'), indent=2)
# json.dump(result2, open(f"./result/{target}_{predict}.json", 'w'))