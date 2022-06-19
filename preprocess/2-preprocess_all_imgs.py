from PIL import Image, ImageEnhance
import  numpy as np
import glob
import os
from tqdm import tqdm


def max_cut(img):
    # cut images white edges.
    # detect the character in the image and crop the character part
    left, right, up, down = 10000, -1, 10000, -1
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            if img[i, j] == 0:
                left = min(left, j)
                up = min(up, i)
                right = max(right, j)
                down = max(down, i)
    assert left>=0 and right>=0 and up>=0 and down>=0
    if left >= right or up>=down:
        return img
    return img[up:down, left:right]

def enhance(path):
    img = Image.open(path).convert("L")
    img = np.array(img)
    img = max_cut(img)
    return Image.fromarray(img).convert("L")


data_path = "/data_local/zhangao/data/oracle/oracle/data/"
path = os.path.join(data_path, "*", "*", "*.png")
# path = "query_list/*.png"
l = glob.glob(path)
for p in tqdm(l):
    img = enhance(p)
    img.save(p)
