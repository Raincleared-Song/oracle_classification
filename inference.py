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
import cv2 as cv

device = "cuda:2"

class SelfMap(nn.Module):
    def __init__(self):
        super(SelfMap, self).__init__()
    def forward(self, x):
        return x




path = ""
model = models.resnet50(pretrained=False)
# model.fc = nn.Sequential(nn.Linear(2048, 2048), nn.ReLU(), nn.Linear(2048, 128))

checkpoint = torch.load("./runs/Nov24_14-36-53_node3/checkpoint_0200.pth.tar", "cpu")
state_dict = {k.replace("backbone.", ""):v for k, v in checkpoint["state_dict"].items()}
log = model.load_state_dict(state_dict, strict=False)
print(log)
model.fc = SelfMap()
model.to(device)

transform = transforms.ToTensor()


def get_img(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)

model.eval()

query_features = []
query_list = glob.glob("./data/binary_query_list/*.png")
# query_list = query_list[:100]
with torch.no_grad():
    for path in tqdm(query_list):
        img = get_img(path).to(device)
        feat = model(img).squeeze(0)
        query_features.append(feat)
query_features = torch.stack(query_features, 0)
query_features = query_features / query_features.norm(dim=-1, keepdim=True)

file_index = 0
# load all image features
l = json.load(open(f"./orcal/all_file_names.json"))
features = torch.load(f"./orcal/all_feats.pth").to(device)

print(query_features.size(), features.size())
similarity = torch.matmul(features, query_features.T)
print(similarity.size())
with open('./orcal/orcal_query_name_list.json','w') as f:
    json.dump(query_list, f)

torch.save(similarity, './orcal/all_similarity.pth')
'''
for k in tqdm(range(similarity.size(0))):
    sims = similarity[k]
    # filter out unsimilar samples
    flag_ = sims.topk(500,sorted=True)
    values = flag_.values.cpu().numpy().tolist()
    flag = flag_.indices.cpu().numpy().tolist()
    #retained_idxs = torch.where(flag)[0]
    
    #sims = sims[flag]
    #values, idxs = sims.sort(descending=True)
    #idxs = retained_idxs[idxs]
    sorted_l = [(query_list[i].split('/')[-1], val) for i, val in zip(flag, values)]

    det_dir = os.path.join(f"/home/chenqianyu/orcal/ranked_images/{file_index}", os.path.basename(l[k]))
    os.makedirs(det_dir, exist_ok=True)
    os.makedirs(det_dir+'/ranked_imgs', exist_ok=True)
    img = cv.imread(l[k])
    cv.imwrite(det_dir+'/'+os.path.basename(l[k]), img)
    # sorted_l = [(l[i], float(val)) for i, val in zip(idxs, values) if val > 0.8]
    for i, (path, score) in enumerate(sorted_l):
        img = cv.imread('/home/chenqianyu/SimCLR-release/data/binary_query_list/'+path)
        cv.imwrite(det_dir+f"/ranked_imgs/{i}_"+path,img)
        #det_name = str(i) + "_" + str(round(float(score), 4)) + ".png"
        #shutil.copy(path, os.path.join(det_dir, det_name))
'''