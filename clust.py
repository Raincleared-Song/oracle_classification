import sys, os
from sklearn.manifold import TSNE
from cuml.cluster import KMeans, DBSCAN
# from sklearn.cluster import DBSCAN
import numpy as np
import cudf as pd

import matplotlib.pyplot as plt
import pickle
import json

print(sys.argv)
# char = sys.argv[1]
#
# if char == "all":
#     prefix = "handa_all"
# else:
#     prefix = f"handa_{char}"
prefix = sys.argv[1]
feats_path = f"./feats/{prefix}_feats.npy"
print(f"load feats from {feats_path}")
X = np.load(feats_path)
print(X.shape)
df = pd.DataFrame(X)
eps = float(sys.argv[2])
# y = KMeans(n_clusters=4015).fit_predict(X)
y = DBSCAN(eps=eps, min_samples=2).fit_predict(X)
labels = np.unique(y)
print(f"Has {len(labels)} labels")
# if char != "all":
#     import seaborn as sns
#     pkl_file = f"./feats/{prefix}_feats_tsne.pkl"
#     if os.path.exists(pkl_file):
#         tsne_results = pickle.load(open(pkl_file, 'rb'))
#     else:
#         tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, n_jobs=-1)
#         tsne_results = tsne.fit_transform(X)
#         with open(pkl_file, 'wb') as f:
#             pickle.dump(tsne_results, f)
#     df['y'] = y
#     df['tsne-2d-one'] = tsne_results[:,0]
#     df['tsne-2d-two'] = tsne_results[:,1]
#     plt.figure(figsize=(16,10))
#     sns.scatterplot(
#         x="tsne-2d-one", y="tsne-2d-two",
#         palette=sns.color_palette("hls", len(labels)),
#         data=df,
#         hue='y',
#         legend="auto",
#         alpha=0.3
#     )
#     if not os.path.exists("imgs"):
#         os.mkdir("imgs")
#     plt.savefig(f"imgs/{prefix}_tsne.jpg")

# if len(sys.argv) > 3 and sys.argv[3] == "label_path":
paths = json.load(open(f"feats/{prefix}_paths.json", 'r'))
label2path = dict.fromkeys(list(map(str, labels)))
for i, label in enumerate(y):
    label = str(label)
    if label2path.get(label) is None:
        label2path[label] = []
    label2path[str(label)].append(paths[i])
single_cnt = 0
for label, paths in label2path.items():
    if len(paths) > 50:
        print(f"label {label}: {len(paths)}")
    elif len(paths) == 1:
        single_cnt += 1
print("singe_cnt =", single_cnt)

json.dump(label2path, open(f'feats/{prefix}_dbscan_eps={eps}_label2path.json', 'w'),indent=2)