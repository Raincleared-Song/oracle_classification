import os

import cv2
import glob

def binary(imgDir):
    name = imgDir.split(".")[0]
    fmt = imgDir.split(".")[1]
    img = cv2.imread(imgDir, 0)
    # img = cv2.medianBlur(img, 5)  # 中值滤波
    # 自适应阈值二值化
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 1)
    img = cv2.medianBlur(img, 5)  # 中值滤波
    #cv2.imshow("img", img)
    cv2.imwrite(name + "_cv." + fmt, img)
    print(imgDir + " process successfully!", end='\r', flush=True)


base_dirs = ["/home/zhangzeyuan/zeyuan/supervised_data/data_1", "/home/zhangzeyuan/zeyuan/supervised_data/data_2"]
dir = base_dirs[0]
# for item in os.listdir("/home/zhangzeyuan/zeyuan/supervised_data/data_1"):
#     for image in glob.glob(f'{dir}/{item}/*.jpg'):
#         binary(image)
dir = base_dirs[1]
for item in os.listdir(dir):
    for image in glob.glob(f'{dir}/{item}/*.png'):
        binary(image)
