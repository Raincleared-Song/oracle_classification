import json
from pathlib import Path
import os
import shutil
label2path = json.load(open("/home/zhangzeyuan/zeyuan/SimCLR-release/feats/handa_8251_dbscan_label2path.json", 'r'))
root_path = "/home/zhangzeyuan/zeyuan/handa/8251"
shutil.rmtree(root_path, ignore_errors=True)
Path(root_path).mkdir(parents=True)
for label, paths in label2path.items():
    label_dir = f"{root_path}/{label}"
    os.mkdir(label_dir)
    for path in paths:
        shutil.copy(path, label_dir)