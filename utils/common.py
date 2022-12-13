import logging
import os
import random
import sys

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm


def save_dataset_preview(train_set, std, mean, save_dir):
    plt.figure(figsize=(8, 8))
    for i in range(9):
        img, label = train_set[random.randint(0, len(train_set))]
        for i in range(img.shape[0]):
            img[i] = img[i] * std[i] + mean[i]
        img = img.permute(1, 2, 0)
        ax = plt.subplot(3, 3, i + 1)
        ax.imshow(img.numpy())
        ax.set_title("label = %d" % label)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(save_dir)


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


def calculate_mean_std(image_dir, img_size):
    """
    计算图片数据集的mean和std
    :param image_dir: 图片数据路径
    :return: mean,std
    """
    img_h, img_w = img_size, img_size
    means, stdevs = [], []
    img_list = []
    dirs_list = []
    dirs_list = get_file(image_dir, dirs_list)
    i = 0
    for item in tqdm(dirs_list):
        img = cv2.imread(item)
        img = cv2.resize(img, (img_w, img_h))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)
        i += 1
        # print(i, '/', len_)
    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.
    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
    # BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
    means.reverse()
    stdevs.reverse()
    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    return means, stdevs


def record_mean_std(mean, std):
    """
    记录mean和std的计算结果到txt文件中
    :param mean: mean
    :param std: std
    :return:
    """
    root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    with open(os.path.join(root, "mean_std.txt"), "w")as f:
        f.writelines("{}-{}-{}\n".format(mean[0], mean[1], mean[2]))
        f.writelines("{}-{}-{}\n".format(std[0], std[1], std[2]))


def read_mean_std():
    """
    从txt中读取mean和std
    :return: mean, std
    """
    try:
        root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        with open(os.path.join(root, "mean_std.txt"), "r")as f:
            lines = f.readlines()
            mean = [float(lines[0].split("-")[0]), float(lines[0].split("-")[1]), float(lines[0].split("-")[2])]
            std = [float(lines[1].split("-")[0]), float(lines[1].split("-")[1]), float(lines[1].split("-")[2])]
            return mean, std
    except:
        print("read_mean_std wrong! return default [0.4, 0.4, 0.4], [0.2, 0.2, 0.2]")
        return [0.4, 0.4, 0.4], [0.2, 0.2, 0.2]


def show_data_info(dataset):
    """
    打印dataset的类别、每一类的图片数量、图片总数
    :param dataset:
    :return:
    """
    classes = dataset.classes
    class_dic = dataset.class_to_idx
    target = dataset.targets
    data_info = {}
    print("Classes is {}".format(classes))
    for key in class_dic.keys():
        print("{}:{}".format(key, target.count(class_dic[key])))
        data_info[key] = target.count(class_dic[key])
    print("Total number is:{}".format(len(target)))

def top_N(n, outputs, labels):
    soft_pridict = torch.softmax(outputs, dim=1)
    # 得到前五个的概率和标签
    value, indices = torch.topk(soft_pridict, n)
    top_correct = 0
    if n == 1:
        top_correct = (indices[:, n-1] == labels).sum()
    elif n == 3:
        top_correct += (indices[:, n-1] == labels).sum()
        top_correct += (indices[:, n-2] == labels).sum()
        top_correct += (indices[:, n-3] == labels).sum()
    elif n == 5:
        top_correct += (indices[:, n-1] == labels).sum()
        top_correct += (indices[:, n-2] == labels).sum()
        top_correct += (indices[:, n-3] == labels).sum()
        top_correct += (indices[:, n-4] == labels).sum()
        top_correct += (indices[:, n-5] == labels).sum()

    return top_correct.item() / len(indices)

def add_log_file(infile=None, level=20, backup_count=10):

    logger = logging.getLogger()
    logger.setLevel(level)

    for oneHandler in logger.handlers:
        logger.removeHandler(oneHandler)

    fileHandler = logging.handlers.TimedRotatingFileHandler(filename=infile, when='midnight',
                                                            backupCount=backup_count, encoding='utf-8')
    #fileHandler = logging.FileHandler(infile)
    consoleHandler = logging.StreamHandler(sys.stdout)

    fileHandler.setLevel(level)
    consoleHandler.setLevel(logging.WARNING)

    formatter = logging.Formatter('[%(levelname)s,%(asctime)s %(filename)s:%(lineno)d]%(message)s',
                                  "%m-%d %H:%M:%S.%3M")
    fileHandler.setFormatter(formatter)
    consoleHandler.setFormatter(formatter)

    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)
