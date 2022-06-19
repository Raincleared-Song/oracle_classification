from PIL import Image
import numpy as np
import os
import glob
from tqdm import tqdm
from multiprocessing import Pool
from data.utils import get_handa_classes_data

data_root = "./classes"
data_root2 = "./classes2"

def max_cut(img):
    # cut images white edges.
    # detect the character in the image and crop the character part
    #
    img = np.array(img.convert("L"))
    left, right, up, down = 10000, -1, 10000, -1
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            if img[i, j] > 0: # for black background only
                left = min(left, j)
                up = min(up, i)
                right = max(right, j)
                down = max(down, i)
    if not(left>=0 and right>=0 and up>=0 and down>=0):
        return Image.fromarray(img)
    if left >= right or up>=down:
        return Image.fromarray(img)
    vert_len = down - up
    hori_len = right - left
    width = max(vert_len, hori_len)
    new_img = np.zeros((width, width))
    if vert_len > hori_len:
        hori_start = (vert_len - hori_len) // 2
        vert_start = 0
    else:
        vert_start = (hori_len - vert_len) // 2
        hori_start = 0
    new_img[vert_start:vert_start+vert_len, hori_start:hori_start+hori_len] = img[up:down, left:right]
    return Image.fromarray(new_img).convert("RGB")

def preprocess_img(path):
    img = Image.open(path)
    img = max_cut(img)
    img.save(f'{data_root2}{path[len(data_root):]}')


def preprocess_class_folder(class_folder):
    if not os.path.exists(f'{data_root2}/{class_folder}'):
        os.mkdir(f'{data_root2}/{class_folder}')
    for contrast_ratio in os.listdir(f'{data_root}/{class_folder}/'):
        if not os.path.exists(f'{data_root2}/{class_folder}/{contrast_ratio}'):
            os.mkdir(f'{data_root2}/{class_folder}/{contrast_ratio}')
        for path in glob.glob(f'{data_root}/{class_folder}/{contrast_ratio}/*.png'):
            img = Image.open(path)
            img = max_cut(img)
            img.save(f'{data_root2}{path[len(data_root):]}')



if __name__ == "__main__":

    if not os.path.exists(data_root2):
        os.mkdir(data_root2)
    imgs = []
    for class_folder in os.listdir(data_root):
        if not os.path.exists(f'{data_root2}/{class_folder}'):
            os.mkdir(f'{data_root2}/{class_folder}')
        for contrast_ratio in os.listdir(f'{data_root}/{class_folder}/'):
            if not os.path.exists(f'{data_root2}/{class_folder}/{contrast_ratio}'):
                os.mkdir(f'{data_root2}/{class_folder}/{contrast_ratio}')
            for path in glob.glob(f'{data_root}/{class_folder}/{contrast_ratio}/*.png'):
                imgs.append(path)

    pool = Pool(32)
    with tqdm(total=len(imgs)) as pbar:
        for i, _ in enumerate(pool.imap_unordered(preprocess_img, imgs)):
            pbar.update()
    pool.close()
    pool.join()