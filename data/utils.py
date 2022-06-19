import glob
import os
import json
import pickle

from PIL import Image
import random
from tqdm import tqdm
from data.oracle_db import *
import re
import numpy as np

random.seed(0)

def remove_osx_ds_store(l: list):
    if '.DS_Store' in l:
        l.remove('.DS_Store')
    return l


def get_supervised_data(data_paths: list, formats: list, ratios=(1.0,)):
    # assert(len(data_paths) == len(formats))
    img_parts = [[] for _ in range(len(ratios))]
    empty_part = [0] * len(ratios)
    y = 0
    y2label = {}
    for i, data_path in enumerate(data_paths):
        print(f"loading {data_path}")
        format = formats[i]
        img_dirs = os.listdir(data_path)
        img_dirs = remove_osx_ds_store(img_dirs)
        img_dirs = sorted(img_dirs, key=lambda x: int(x))
        for index in img_dirs:  # index = 0, 1, 2...
            y2label[y] = index
            imgs = glob.glob(f'{data_path}/{index}/*.{format}')
            imgs = [img for img in imgs if "cv" not in img]
            consumed = 0
            for i, ratio in enumerate(ratios):
                part_len = min(max(1, int(len(imgs) * ratio)), len(imgs) - consumed) if ratio > 0 else 0
                if part_len == 0:
                    empty_part[i] += 1
                img_parts[i] += [(img, y) for img in imgs[consumed: consumed+part_len]]
                consumed += part_len
            y += 1
    for i, item in enumerate(empty_part):
        print(f"part {i} has {item} empty classes")
    return (*img_parts, y, y2label)


def get_oracle_paths(data_path: str):
    l = []
    img_dirs = os.listdir(data_path)
    img_dirs = remove_osx_ds_store(img_dirs)
    img_dirs = sorted(img_dirs, key=lambda x: int(x))
    for index in img_dirs:  # index = 0, 1, 2...
        for image in glob.glob(f'{data_path}/{index}/1/*.png'):
            l.append(image)
    return l


def get_standard_paths(base: str, format='png'):
    return glob.glob(f'{base}/*.{format}')

def get_jiaguwenbian_data(data_root: str="./甲骨文字编摹本448字7382样本", ratios=(0.8,0.1,0.1), freq_limit=1e10):
    img_parts = [[] for _ in range(len(ratios))]
    empty_part = [0] * len(ratios)
    class_folders = []
    y2label = {}
    for folder in os.listdir(data_root):
        # if folder in ["B", "D", "c", "unk"]:
        #     continue
        if folder in ["一", "二", "三", "四", "五", "甲3072", "B"]:
            continue
        class_folders.append(folder)
    class_folders.sort()
    ind = 0
    for class_folder in tqdm(class_folders):
        y = ind
        y2label[y] = class_folder
        imgs = sorted(glob.glob(f"{data_root}/{class_folder}/*.png"))
        if len(imgs) == 0 or len(imgs) > freq_limit:
            continue
        consumed = 0
        for i, ratio in enumerate(ratios):
            part_len = min(max(1, int(len(imgs) * ratio)), len(imgs) - consumed) if ratio > 0 else 0
            if part_len == 0:
                empty_part[i] += 1
            img_parts[i] += [(img, y) for img in imgs[consumed: consumed + part_len]]
            consumed += part_len
        ind += 1
    for i, item in enumerate(empty_part):
        print(f"part {i} has {item} empty classes")
    return (*img_parts, len(y2label), y2label)

def get_handa_classes_data(data_root: str="/data/private/songchenyang/hanzi_filter/classes", contrast_ratios = [0], ratios=(0.9, 0.1), relabel=True, get_specific_char=False, freq_limit=1e10, use_all=False, sort_imgs=False):
    img_parts = [[] for _ in range(len(ratios))]
    empty_part = [0] * len(ratios)
    class_folders = []
    y2label = {}
    if get_specific_char:
        class_folders.append(get_specific_char)
    else:
        for folder in os.listdir(data_root):
            if not use_all and ("-" in folder or folder.startswith("8251") or folder.startswith("9632")):
                continue
            class_folders.append(folder)
    class_folders.sort()
    print(f"loading {data_root}")
    ind = 0
    for class_folder in tqdm(class_folders):
        if not relabel:
            y = class_folder
        else:
            y = ind
            y2label[y] = class_folder
        imgs = []
        for contrast_ratio in contrast_ratios:
            folder_imgs = glob.glob(f'{data_root}/{class_folder}/{contrast_ratio}/*.png')
            if sort_imgs:
                folder_imgs = sorted(folder_imgs)
            imgs += folder_imgs
        if len(imgs) == 0:
            continue
        if len(imgs) > freq_limit: # filter long tail examples
            continue
        consumed = 0
        for i, ratio in enumerate(ratios):
            part_len = min(max(1, int(len(imgs) * ratio)), len(imgs) - consumed) if ratio > 0 else 0
            if part_len == 0:
                empty_part[i] += 1
            img_parts[i] += [(img, y) for img in imgs[consumed: consumed + part_len]]
            consumed += part_len
        ind += 1
    for i, item in enumerate(empty_part):
        print(f"part {i} has {item} empty classes")
    y2label_json = f"./data/handa_y2label_{len(y2label)}.json"
    if not os.path.exists(y2label_json):
        json.dump(y2label, open(y2label_json, 'w'))
        print(f"y2label has been saved to {y2label_json}")
    return (*img_parts, len(y2label), y2label)

def get_oracle_labeled_data(label_json: str="./data/oracle1-2k_labels.json", data_root: str="./data", ratios=(0.9,0.1)):
    img_parts = [[] for _ in range(len(ratios))]
    empty_part = [0] * len(ratios)
    label2info = json.load(open(label_json))
    y2char =  {}
    print(f"loading {label_json}")
    y = 0
    for key, info in label2info.items():
        protos = info["proto"]
        imgs = [f"{data_root}/{proto}" for proto in protos] if isinstance(protos, list) else [f"{data_root}/{protos}"]
        consumed = 0
        for i, ratio in enumerate(ratios):
            part_len = min(max(1, int(len(imgs) * ratio)), len(imgs) - consumed) if ratio > 0 else 0
            if part_len == 0:
                empty_part[i] += 1
            img_parts[i] += [(img, y) for img in imgs[consumed: consumed + part_len]]
            consumed += part_len
        y2char[y] = info["char"]
        y += 1
    for i, item in enumerate(empty_part):
        print(f"part {i} has {item} empty classes")
    return (*img_parts, len(label2info), y2char)

def get_handa_category_data(data_root="/data/private/songchenyang/hanzi_filter/handa", ratios=(0.9, 0.1)):
    img_parts = [[] for _ in range(len(ratios))]
    empty_part = [0] * len(ratios)
    category2imgs = {}
    for collection in next(os.walk(data_root))[1]:
        if collection == "extra":
            continue
        character_folder = os.path.join(data_root, collection, 'characters')
        meta_data = json.load(open(os.path.join(data_root, collection, f'oracle_meta_{collection}.json')))
        for item in meta_data:
            l_chars = item["l_chars"]
            if len(l_chars) == 0:
                continue
            category = item["category"]
            if len(category) == 0:
                continue
            if category2imgs.get(category) is None:
                category2imgs[category] = []
            category2imgs[category] += [os.path.join(character_folder,  ch["img"]) for ch in l_chars]
    y2category = {}
    for y, category in enumerate(category2imgs.keys()):
        imgs = category2imgs[category]
        y2category[y] = category
        consumed = 0
        for i, ratio in enumerate(ratios):
            part_len = min(max(1, int(len(imgs) * ratio)), len(imgs) - consumed) if ratio > 0 else 0
            if part_len == 0:
                empty_part[i] += 1
            img_parts[i] += [(img, y) for img in imgs[consumed: consumed + part_len]]
            consumed += part_len
    for i, item in enumerate(empty_part):
        print(f"part {i} has {item} empty classes")
    return (*img_parts, len(category2imgs), y2category)

def get_oracle_db_data(data_root: str="ocr_char", ratios=(0.9, 0.1), freq_limit=1e10, relabel=False, returnPath2Id=False, y2label=None):
    print(f"loading handa_oracle.db")
    init_db()
    img_parts = [[] for _ in range(len(ratios))]
    empty_part = [0] * len(ratios)
    if y2label is None:
        y2label = json.load(open("./data/handa_y2label_3297.json", 'r'))
    label2y = {v: int(k) for k, v in y2label.items()}
    id2imgs = {}
    not_exist_cnt = 0
    id2char = {}
    for charshape in CharFace.select().where(CharFace.match_case==2): # 7704
        if id2imgs.get(charshape.liding_character) is None:
            id2imgs[charshape.liding_character] = []
        path = f"{data_root}/{charshape.wzb_handcopy_face}"
        if os.path.exists(path):
            id2imgs[charshape.liding_character].append(path)
        else:
            not_exist_cnt += 1
    print(f"{not_exist_cnt} files not exist")

    ch_not_in_handa_cnt = 0
    img_id = 0
    path2id = {}
    for id, imgs in id2imgs.items():
        ch = Character.get(id=id)
        label = str(ord(ch.modern_character))
        id2char[id] = label
        if len(ch.chant_font_label) > 0:
            label += "+" + ch.chant_font_label
        y = label2y.get(label)
        if y is None:
            ch_not_in_handa_cnt += 1
            continue
        if len(imgs) > freq_limit: # filter long tail examples
            continue
        consumed = 0
        for i, ratio in enumerate(ratios):
            part_len = min(max(1, int(len(imgs) * ratio)), len(imgs) - consumed) if ratio > 0 else 0
            if part_len == 0:
                empty_part[i] += 1
            if relabel:
                for img in imgs[consumed: consumed + part_len]:
                    if path2id.get(img) is None:
                        path2id[img] = id
                        img_parts[i].append((img, img_id))
                        img_id += 1
                    else:
                        print(f"{path} has more than one Id: {path2id[img]}, {id}")

            else:
                img_parts[i] += [(img, y) for img in imgs[consumed: consumed + part_len]]
            consumed += part_len
    print(f"{ch_not_in_handa_cnt} characters not in handa classes") # 54
    save_path = f"./data/oracle_db_id2char_{len(id2char)}.json"
    json.dump(id2char, open(save_path, 'w'))
    print(f"saving id2char to {save_path}")
    for i, item in enumerate(empty_part):
        print(f"part {i} has {item} empty classes")
    res = (*img_parts, len(y2label), y2label)
    if returnPath2Id:
        res = (*res, path2id)
    return res

def get_oracle_tokens(data_root="./oracle_token"):
    res = []
    pattern = re.compile(r'\d\d\d\d')
    for folder in os.listdir(data_root):
        if not pattern.match(folder):
            continue
        for name in os.listdir(f"{data_root}/{folder}"):
            attrs = name[:-4].split("_")
            if attrs[7] == "0":
                res.append((f"{data_root}/{folder}/{name}", folder))
    return res

def catGeneratedImage(path: str):
    start = path.find("classes/") + len("classes/")
    img1 = Image.open(path).convert("RGB")
    img2 = Image.open(f"/home/hx/zeyuan/handa_classes/{path[start:]}").convert("RGB")
    new_im = Image.new('RGB', (img1.width + img2.width, img1.height))
    new_im.paste(img1, (0, 0))
    new_im.paste(img2, (0, img1.width))
    return new_im
