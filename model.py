import os.path
import pickle

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
from torchvision.transforms import transforms
from tqdm import tqdm
import json
from utils import euclidean_dist, Counter, accuracy, get_feats
import numpy as np

class DualModel(nn.Module):
    def __init__(self, checkpoint_path, pred_dataloader, cache_path, use_best_feat=False, device=None):
        super(DualModel, self).__init__()
        self.device = device
        self.backbone = models.resnet50(pretrained=False)
        self.backbone.fc = nn.Identity()
        checkpoint = torch.load(checkpoint_path, "cpu")
        state_dict = checkpoint["state_dict"]
        self.backbone.load_state_dict(state_dict, strict=False)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.out_dim = state_dict["fc.weight"].shape[0]
        self.linear = torch.nn.Linear(2048, self.out_dim)
        self.linear.weight = torch.nn.Parameter(state_dict["fc.weight"])
        self.linear.bias = torch.nn.Parameter(state_dict["fc.bias"])
        self.linear.requires_grad_(False)
        self.w = nn.Parameter(torch.ones(self.out_dim))
        self.relu = nn.ReLU()
        # self.w2 = nn.Parameter(torch.zeros(self.out_dim))
        self.to(self.device)
        self.pred_feats, self.y2ind= get_feats(self.model, pred_dataloader, device, cache_path)
        self.ind2y = {v: k for k, v in self.y2ind.items()}
        #
        # if not os.path.exists("./cache"):
        #     os.mkdir("./cache")
        # if cache_path is not None:
        #     cache_path = "./cache" + "/" + cache_path
        #     if use_best_feat:
        #         cache_path += "_best_feat"
        #     cache_path += ".pkl"
        # if cache_path is not None and os.path.exists(cache_path):
        #     self.pred_feats, self.y2ind = pickle.load(open(cache_path, 'rb'))
        #     self.ind2y = {v: k for k, v in self.y2ind.items()}
        # elif pred_dataloader is not None:
        #     self.pred_feats, self.ind2y = self.get_feats(pred_dataloader, use_best_feat=use_best_feat)
        #     self.y2ind = {k: v for v, k in self.ind2y.items()}
        #     pickle.dump((self.pred_feats.cpu(), self.y2ind), open(cache_path, 'wb'))
        self.pred_feats = nn.Parameter(self.pred_feats.to(device), requires_grad=False)
        # self.decoder = nn.Sequential(nn.Linear(self.out_dim+self.pred_feats.shape[0], self.out_dim, bias=False), nn.Softmax(dim=1))
        # print(self.decoder[0].weight.shape)
        # torch.nn.init.eye_(self.decoder[0].weight[:, :self.out_dim])
        # torch.nn.init.zeros_(self.decoder[0].weight[:, self.out_dim:])



    def forward(self, x_batch):
        with torch.no_grad():
            feats = self.backbone(x_batch)
            p_y_linear = F.softmax(self.linear(feats), dim=1)
            p_y_proto = F.softmax(-euclidean_dist(feats, self.pred_feats), dim=1)
        w = self.relu(self.w)
        p_y = w*p_y_linear + (1-w)*p_y_proto
        return p_y

    def trainable_parameters(self):
        return {'w': self.w.cpu()}

    def load_parameters(self, state_dict):
        self.w = state_dict['w']

    def get_feats(self, dataloader, y2label=None, use_best_feat=False):
        label2feats = {}
        label2p = {}
        for x_batch, y_batch in tqdm(dataloader):
            with torch.no_grad():
                feats = self.backbone(x_batch.to(self.device))
                if use_best_feat:
                    p_y = F.softmax(self.linear(feats), dim=1)
                for i, feat in enumerate(feats):
                    label = int(y_batch[i])
                    if y2label is not None:
                        label = y2label[label]
                    if label2feats.get(label) is None:
                        label2feats[label] = []
                    if use_best_feat:
                        p = float(p_y[i][y_batch[i]])
                        if label2p.get(label, 0) < p:
                            label2p[label] = p
                            label2feats[label] = [feat]
                    else:
                        label2feats[label].append(feat)
        mean_feats = torch.empty((len(label2feats.keys()), 2048), device=self.device, dtype=torch.float)
        labels = sorted(list(label2feats.keys()))
        ind2label = {}
        for i, y in enumerate(labels):
            feats = label2feats[y]
            mean_feats[i] = sum(feats)/len(feats)
            ind2label[i] = y

        return mean_feats, ind2label

class OCRModel:
    def __init__(self,
                 checkpoint_path="../SimCLR-release/checkpoints/handa_best.pth.tar",
                 pred_dataloader=None,
                 paths_json=None,
                 topk=(1,5),
                 device="cuda:0",
                 use_best_feat=False,
                 cache_path=None,
                 threshold=0.65,
                 y2label=None):
        self.device = device
        self.topk = topk
        self.model = models.resnet50(pretrained=False)
        self.model.fc = torch.nn.Identity()
        checkpoint = torch.load(checkpoint_path, "cpu")
        state_dict = checkpoint["state_dict"]
        print(self.model.load_state_dict(state_dict, strict=False))
        self.linear = torch.nn.Linear(2048, state_dict["fc.weight"].shape[0])
        self.linear.weight = torch.nn.Parameter(state_dict["fc.weight"])
        self.linear.bias = torch.nn.Parameter(state_dict["fc.bias"])
        self.linear.to(device)
        self.model.to(device)
        self.eval_ = False
        if pred_dataloader is not None:
            self.pred_feats, self.y2ind = get_feats(self.model, pred_dataloader, device, cache_path)
            self.ind2y = {v: k for k, v in self.y2ind.items()}
        else:
            self.pred_feats = None
        self.pred_feats = self.pred_feats.to(device)
        print(self.pred_feats.shape)
        if paths_json:
            self.pred_paths = json.load(open(paths_json, 'r'))
        self.maxk = max(self.topk)
        # self.eval()
        self.threshold = threshold
        self.y2label = y2label
        self.log = open(f"log/ocr_result.txt", 'w')

    def eval(self):
        self.eval_ = True
        self.model.eval()
        self.thresholds = [0+0.05 * i for i in range(21)]
        self.counters = {t:[Counter() for k in self.topk] for t in self.thresholds}
        self.classCounters = {t:[dict() for k in self.topk] for t in self.thresholds}

    def __call__(self, x_batch, y_batch=None):
        x_batch = x_batch.to(self.device)
        feats = self.model(x_batch)
        p_y_linear = F.softmax(self.linear(feats), dim=1)
        top_p, pred = p_y_linear.topk(self.maxk, 1, True, True)
        top_p_sum = torch.sum(top_p[:, :5], dim=1, keepdim=True)
        if self.pred_feats is not None:
            dists = euclidean_dist(feats, self.pred_feats)
            p_y_proto = F.softmax(-dists, dim=1)
            _, pred2 = p_y_proto.topk(self.maxk, 1, True, True)
            #Note: can skip identical mapping if feats indices are one-to-one against ys
            pred2 = pred2.cpu().apply_(lambda x: self.ind2y[x]).to(self.device)
        # if not self.eval_:
        #     correct = pred[:, 1].eq(y_batch)
        #     pred2 = pred * (top_p_sum > self.threshold) + pred2 * (top_p_sum <= self.threshold)
        #     correct2 = pred2[:, 1].eq(y_batch)
        #     for i, y in enumerate(y_batch.cpu().tolist()):
        #         if not correct[i] and correct2[i]:
        #             self.log.write(f"{self.y2label[y]}\t{self.y2label[int(pred[i][0])]}\t{self.y2label[int(pred2[i][0])]}")
        #     pred = pred2
        if y_batch is not None:
            y_batch = y_batch.to(self.device)
            for t in self.thresholds:
                pred = pred * (top_p_sum>t) + pred2 * (top_p_sum<=t)
                correct = pred.eq(y_batch.view(-1, 1).expand_as(pred))
                for i, k in enumerate(self.topk):
                    correct_k = torch.sum(correct[:, :k].float(), dim=1)
                    self.counters[t][i].add(float(torch.sum(correct_k)), int(y_batch.size(0)))
                    for j, y in enumerate(y_batch.cpu().tolist()):
                        # if self.y2label[y] == "20127":
                        #     print(top_p[j].tolist())
                        #     print(float(top_p_sum[j]))
                        if self.classCounters[t][i].get(y) is None:
                            self.classCounters[t][i][y] = Counter()
                        self.classCounters[t][i][y].add(correct_k[j])
        return pred

    def report(self):
        for i, k in enumerate(self.topk):
            print(f"Threshold\t top{k} acc\t class acc")
            for t in self.thresholds:
                class_accs = [counter.average() for counter in self.classCounters[t][i].values()]
                n_zero_accs = Counter()
                for acc in class_accs:
                    if acc > 0:
                        n_zero_accs.add(acc)
                # print(f"none_zero_cnt={len(n_zero_accs)}; none_zero_avg={n_zero_accs.average()}; zero_cnt={len(class_accs)-len(n_zero_accs)}")
                class_acc = sum(class_accs)/len(class_accs)
                print(f"{t:.2f}\t{self.counters[t][i].average()*100}\t{class_acc*100}")
                # print(f"{t:.2f}\t{self.counters[t][i].average()}")
        i=0
        for t in [0,0.55, 0.9]:
            f = open(f"./log/t={t}_class_acc.txt", 'w')
            for y, counter in self.classCounters[t][i].items():
                # print(f"{y}: {counter.sum()}/{len(counter)}")
                f.write(f"{self.y2label[y]}: {counter.sum()}/{len(counter)}\n")
            f.close()

    def get_feats(self, dataloader, use_best_feat=False):
        self.model.eval()
        y2feats = {}
        y2p = {}
        for x_batch, y_batch in tqdm(dataloader):
            with torch.no_grad():
                feats = self.model(x_batch.to(self.device))
                if use_best_feat:
                    p_y = F.softmax(self.linear(feats), dim=1)
                for i, feat in enumerate(feats):
                    y = int(y_batch[i])
                    if y2feats.get(y) is None:
                        y2feats[y] = []
                    if use_best_feat:
                        p = float(p_y[i][y])
                        if y2p.get(y, 0) < p:
                            y2p[y] = p
                            y2feats[y] = [feat]
                    else:
                        y2feats[y].append(feat)
        mean_feats = torch.empty((len(y2feats), 2048), device=self.device, dtype=torch.float)
        ys = sorted(list(y2feats.keys()))
        ind2y = {}
        for i, y in enumerate(ys):
            feats = y2feats[y]
            mean_feats[i] = sum(feats)/len(feats)
            ind2y[i] = y

        return mean_feats, ind2y
