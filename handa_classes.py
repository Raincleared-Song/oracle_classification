import os
import re
import cv2
import json
import shutil
import numpy as np
from tqdm import tqdm


def main():
    # 将单字数据进行分类
    appear_set = set()
    os.makedirs('classes', exist_ok=True)
    handa_parts = ['B', 'D', 'H', 'L', 'S', 'T', 'W', 'Y', 'HD']
    for part in handa_parts:
        with open(f'/home/hx/zeyuan/handa/{part}/oracle_meta_{part}.json', 'r', encoding='utf-8') as fin:
            meta = json.load(fin)
        for book in tqdm(meta, desc=part):
            for char in book['l_chars']:
                ch = char['char']
                font = re.findall(r"<font face='([^']+)'>", ch)
                font = list(set(font))
                assert len(font) <= 1
                ch = re.sub(r'</?[^>]+>|[ ]', '', ch)
                assert len(re.findall(r'[/<>a-zA-Z0-9]', ch)) == 0
                folder_name = '-'.join(str(ord(c)) for c in ch)
                if len(font) > 0:
                    folder_name += '+' + font[0]
                if folder_name not in appear_set:
                    appear_set.add(folder_name)
                    os.makedirs('classes/' + folder_name, exist_ok=False)
                # 查看有多少白色像素
                img = cv2.imread(f'/home/hx/zeyuan/handa/{part}/characters/{char["img"]}', flags=0)
                # 阈值暂时卡在 240
                ratio = int(np.sum(img >= 240) * 10 / img.size) * 10
                # 存在全白的情况 B07582-1-521.png 卜
                assert ratio % 10 == 0 and 0 <= ratio // 10 <= 10
                os.makedirs(f'classes/{folder_name}/{ratio}', exist_ok=True)
                shutil.copy(f'/home/hx/zeyuan/handa/{part}/characters/{char["img"]}', f'classes/{folder_name}/{ratio}')
    print(ord('※'), ord('■'))  # 8251 9632


if __name__ == '__main__':
    main()
