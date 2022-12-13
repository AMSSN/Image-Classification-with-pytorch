from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from model_zoo.gen_model import get_efficientnetb0, change_fc
import os, sys, time
import torch
from torch import nn
from utils.common import calculate_mean_std, record_mean_std

FILE = Path(__file__).resolve()  # 当前文件的绝对路径
ROOT = FILE.parents[0]  # 当前文件所在的绝对路径


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_dataset(train_dir, valid_dir, img_size, batch_size):
    """
    :param train_dir: 训练集路径
    :param valid_dir: 验证集路径
    :param img_size: 图片尺寸（resize）
    :param batch_size: 批次大小
    :return: train_set, valid_set（DataLoader）
    """
    trainmean, trainstd = calculate_mean_std(train_dir, img_size)
    # validmean, validstd = calculate_mean_std(valid_dir, img_size)
    record_mean_std(trainmean, trainstd)
    # 准备数据集
    train_augs = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomChoice([transforms.RandomHorizontalFlip(),
                                 transforms.RandomRotation(30),
                                 transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1)]),
        transforms.ToTensor(),
        transforms.Normalize(trainmean, trainstd)
    ])
    valid_augs = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(trainmean, trainstd)
    ])
    train_iter = datasets.ImageFolder(train_dir, transform=train_augs)
    valid_iter = datasets.ImageFolder(valid_dir, transform=valid_augs)
    train_set = DataLoader(train_iter, batch_size=batch_size, shuffle=True)
    valid_set = DataLoader(valid_iter, batch_size=batch_size, shuffle=True)
    return train_set, valid_set


def train_model(train_set, valid_set, net, optimizer, loss_fn, total_epoch, device):
    # 开始训练
    net = net.to(device)
    for epoch in range(1, total_epoch + 1):
        start = time.time()
        net.train()  # 训练模式
        # 一个epoch的损失、准确率、图片数量、批次数
        train_loss_sum, train_acc_sum, n, batch_count = 0.0, 0.0, 0, 0
        for image, label in train_set:
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()  # 梯度清零
            label_hat = net(image)
            loss = loss_fn(label_hat, label)
            loss.backward()
            optimizer.step()
            # 每10个epoch做一次验证（后续更新实时绘图，实时tensorboard）
            train_loss_sum += loss.cpu().item()
            train_acc_sum += (label_hat.argmax(dim=1) == label).sum().cpu().item()
            n += label.shape[0]
            batch_count += 1
        with torch.no_grad():
            net.eval()  # 评估模式
            valid_acc_sum, n2, best_valid_acc = 0.0, 0, 0
            for x, y in valid_set:
                valid_acc_sum += (net(x.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                n2 += y.shape[0]
            # 保存验证集准确率最高的模型
            if valid_acc_sum > best_valid_acc:
                best_valid_acc = valid_acc_sum
                torch.save(net.state_dict(),
                           "./model_weight/train{}_{:.3f}.pth".format(epoch, best_valid_acc / n2))

        print('epoch %d, loss %.8f, train acc %.3f, valid acc %.3f, time %.1f sec'
              % (epoch, train_loss_sum / batch_count, train_acc_sum / n, valid_acc_sum / n2, time.time() - start))
    torch.save(net.state_dict(), "./model_weight/train_final.pth")


if __name__ == '__main__':
    train_set, valid_set = get_dataset('./data_set/train', './data_set/valid', 96, 256)
    net = get_efficientnetb0(False, 'model_weight/efficientnet-b0-355c32eb.pth')
    net = change_fc(net, "efficientnet", 2)
    # optim = torch.optim.SGD(model.parameters(), lr=0.0001)
    # optim = torch.optim.SGD((model.parameters()), lr=0.01, momentum=0.9, weight_decay=0.0004)
    # optim = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9)
    optim = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.99))
    # optim = torch.optim.Adagrad(model.parameters(), lr=0.0001, lr_decay=0.01, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    total_epoch = 1000

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_model(train_set, valid_set, net, optim, loss_fn, total_epoch, device)
