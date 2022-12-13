import os
import shutil
import cv2
import numpy as np


def get_file(root_path, dirs_list):
    """
    递归函数，遍历该文档目录和子目录下的所有文件，获取其path
    :param root_path:
    :param all_files:
    :return:
    """

    image_suffix = ['png', 'jpg', 'jpeg', 'tiff', 'webp', 'svg', 'ico', 'bmp']
    for file in os.listdir(root_path):
        a_file = os.path.join(root_path, file)
        if not os.path.isdir(a_file):  # not a dir
            if a_file.split(".")[-1] in image_suffix:
                dirs_list.append(a_file)
        else:  # is a dir
            get_file(a_file, dirs_list)
    return dirs_list


def rotate_bound_black_bg(img_dir, angle):
    img_ = cv2.imread(img_dir)
    # center
    (h, w) = img_.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # borderValue 缺失背景填充色彩，此处为白色，可自定义
    rotate_img = cv2.warpAffine(img_, M, (nW, nH), borderValue=(0, 0, 0))
    # borderValue 缺省，默认是黑色（0, 0 , 0）
    r_img = img_dir.split(".jpg")[0] + '_r' + str(angle) + '.jpg'
    cv2.imwrite(r_img, rotate_img)


def gaussian(img_dir):
    img_ = cv2.imread(img_dir)
    dst = cv2.GaussianBlur(img_, (5, 5), 0)
    r_img = img_dir.split(".jpg")[0] + '_gauss.jpg'
    cv2.imwrite(r_img, dst)


def median(img_dir):
    img_ = cv2.imread(img_dir)
    dst = cv2.medianBlur(img_, 5)
    r_img = img_dir.split(".jpg")[0] + '_demian.jpg'
    cv2.imwrite(r_img, dst)


def clean_img(img_dir):
    if img_dir.endswith('_demian.jpg'):
        os.remove(img_dir)
        # print(img_dir)
    if img_dir.endswith('_gauss.jpg'):
        os.remove(img_dir)
        # print(img_dir)
    # if not img_dir.endswith('_demian.jpg'):
    #     print(img_dir)
    # if not img_dir.endswith('_demian.jpg'):
    #     print(img_dir)


if __name__ == '__main__':
    # img_list = []
    data_dir = r'data_set/train/not_dog'
    # img_list = get_file(data_dir, get_file)
    print(len(os.listdir(data_dir)))
    for img in os.listdir(data_dir):
        img_dir = os.path.join(data_dir, img)
        rotate_bound_black_bg(img_dir, 30)
        rotate_bound_black_bg(img_dir, 330)
        rotate_bound_black_bg(img_dir, 90)
        rotate_bound_black_bg(img_dir, 270)
        gaussian(img_dir)
        median(img_dir)
        # clean_img(img_dir)
    print(len(os.listdir(data_dir)))

