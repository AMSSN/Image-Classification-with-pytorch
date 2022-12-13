import PIL
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import torch.nn as nn
from model_zoo.gen_model import get_efficientnetb0, load_my_weight
import os
from utils.common import read_mean_std, show_data_info, top_N

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def prediction_for_each_image():
    imgdir = "prediction/imgs"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_efficientnetb0(False)
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, 2)
    state_dict = torch.load("./model_weight/train3_0.857.pth")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # 评估模式
    for img in os.listdir(imgdir):
        img_path = os.path.join(imgdir, img)
        transform_valid = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(96),
            transforms.Normalize([0.466, 0.446, 0.426], [0.246, 0.243, 0.248])])
        PIL_img = PIL.Image.open(img_path)
        tensor_img = transform_valid(PIL_img).unsqueeze(0)  # 拓展维度
        tensor_img = tensor_img.to(device)
        outputs = model(tensor_img)
        # 输出概率最大的类别
        _, indices = torch.max(outputs, 1)
        percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        perc = percentage[int(indices)].item()
        print('img_name:{}\tpredicted:{}'.format(img_path, indices.tolist()[0]))


def prediction_for_information(net, img_dir, batch_size, img_size):
    test_mean, test_std = read_mean_std()
    # 准备数据集
    test_augs = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize(test_mean, test_std)
    ])
    test_iter = datasets.ImageFolder(img_dir, transform=test_augs)
    test_set = DataLoader(test_iter, batch_size=batch_size, shuffle=True)
    show_data_info(test_iter)
    # 进行预测
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    with torch.no_grad():
        test_loss_sum, test_acc_sum, n, batch_count = 0.0, 0.0, 0, 0
        for image, label in test_set:
            net.eval()
            image, label = image.to(device), label.to(device)
            predict = net(image)
            # 损失、top1、top3、top5、
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(predict, label)
            test_loss_sum += loss.cpu().item()
            test_acc_sum += (predict.argmax(dim=1) == label).sum().cpu().item()
            n += label.shape[0]
            batch_count += 1
            topn = top_N(1, predict, label)
        print('loss %.8f, Top1 acc %.3f' % (test_loss_sum / batch_count, test_acc_sum / n))


if __name__ == '__main__':
    img_dir = r'./data_set/test'
    net = get_efficientnetb0(False)
    net = load_my_weight(net, "efficientnet", 2, "./model_weight/train875.pth")
    prediction_for_information(net, img_dir, batch_size=7, img_size=96)

    # prediction_for_each_image()

