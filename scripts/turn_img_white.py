from PIL import Image
import numpy as np
import os
import glob
from tqdm import tqdm
from multiprocessing import Pool
from data.utils import get_handa_classes_data

data_root = "../handa_classes"
data_root2 = "../handa_classes_white"

def turn_img_white(img):
    img = img.convert("L")
    img = np.array(img)
    if len(img[img>250]) < img.size * 0.5:
        img = 255 - img
    return Image.fromarray(img).convert("RGB")

def preprocess_img(path):
    img = Image.open(path)
    img = turn_img_white(img)
    img.save(f'{data_root2}{path[len(data_root):]}')


def preprocess_class_folder(class_folder):
    if not os.path.exists(f'{data_root2}/{class_folder}'):
        os.mkdir(f'{data_root2}/{class_folder}')
    for contrast_ratio in os.listdir(f'{data_root}/{class_folder}/'):
        if not os.path.exists(f'{data_root2}/{class_folder}/{contrast_ratio}'):
            os.mkdir(f'{data_root2}/{class_folder}/{contrast_ratio}')
        # for path in glob.glob(f'{data_root}/{class_folder}/{contrast_ratio}/*.png'):
        #     img = Image.open(path)
        #     img = max_cut(img)
        #     img.save(f'{data_root2}{path[len(data_root):]}')



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

    pool = Pool(64)
    with tqdm(total=len(imgs)) as pbar:
        for i, _ in enumerate(pool.imap_unordered(preprocess_img, imgs)):
            pbar.update()
    pool.close()
    pool.join()