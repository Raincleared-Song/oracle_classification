from PIL import Image, ImageEnhance
import  numpy as np
import glob
import os
from tqdm import tqdm


def enhance(path):
    # binarize images: if a pixel > 127.5, convert it to 255, othervise convert it to 0
    img = Image.open(path).convert("L")
    img = ImageEnhance.Contrast(img).enhance(1000)
    img = np.array(img)
    img[(img>127.5)&(img<255)] = 255
    img[(img<=127.5)&(img<255)] = 0
    return Image.fromarray(img).convert("L")


# data_path = "/data_local/zhangao/data/oracle/oracle/data/"
# path = os.path.join(data_path, "*", "*", "*.png")
path = "query_list/*.png"
l = glob.glob(path)
for p in tqdm(l):
    img = enhance(p)
    img.save(p)
