import torch
from torch import nn

from model_zoo.efficientnet_v1.model import EfficientNet
from model_zoo.resnet.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


def get_efficientnetb0(load_official_weight, load_local_weight=None):
    model = EfficientNet.from_name("efficientnet-b0")
    if load_official_weight:
        model = EfficientNet.from_pretrained("efficientnet-b0")
    if load_local_weight is not None:
        state_dict = torch.load(load_local_weight)
        model.load_state_dict(state_dict)
    return model


def get_efficientnetb1(load_official_weight, load_local_weight=None):
    model = EfficientNet.from_name("efficientnet-b1")
    if load_official_weight:
        model = EfficientNet.from_pretrained("efficientnet-b1")
    if load_local_weight is not None:
        state_dict = torch.load(load_local_weight)
        model.load_state_dict(state_dict)
    return model


def get_efficientnetb2(load_official_weight, load_local_weight=None):
    model = EfficientNet.from_name("efficientnet-b2")
    if load_official_weight:
        model = EfficientNet.from_pretrained("efficientnet-b2")
    if load_local_weight is not None:
        state_dict = torch.load(load_local_weight)
        model.load_state_dict(state_dict)
    return model


def get_efficientnetb3(load_official_weight, load_local_weight=None):
    model = EfficientNet.from_name("efficientnet-b3")
    if load_official_weight:
        model = EfficientNet.from_pretrained("efficientnet-b3")
    if load_local_weight is not None:
        state_dict = torch.load(load_local_weight)
        model.load_state_dict(state_dict)
    return model


def get_efficientnetb4(load_official_weight, load_local_weight=None):
    model = EfficientNet.from_name("efficientnet-b4")
    if load_official_weight:
        model = EfficientNet.from_pretrained("efficientnet-b4")
    if load_local_weight is not None:
        state_dict = torch.load(load_local_weight)
        model.load_state_dict(state_dict)
    return model


def get_efficientnetb5(load_official_weight, load_local_weight=None):
    model = EfficientNet.from_name("efficientnet-b5")
    if load_official_weight:
        model = EfficientNet.from_pretrained("efficientnet-b5")
    if load_local_weight is not None:
        state_dict = torch.load(load_local_weight)
        model.load_state_dict(state_dict)
    return model


def get_efficientnetb6(load_official_weight, load_local_weight=None):
    model = EfficientNet.from_name("efficientnet-b6")
    if load_official_weight:
        model = EfficientNet.from_pretrained("efficientnet-b6")
    if load_local_weight is not None:
        state_dict = torch.load(load_local_weight)
        model.load_state_dict(state_dict)
    return model


def get_efficientnetb7(load_official_weight, load_local_weight=None):
    model = EfficientNet.from_name("efficientnet-b7")
    if load_official_weight:
        model = EfficientNet.from_pretrained("efficientnet-b7")
    if load_local_weight is not None:
        state_dict = torch.load(load_local_weight)
        model.load_state_dict(state_dict)
    return model


def get_resnet18(load_official_weight, load_local_weight=None):
    model = resnet18()
    if load_official_weight:
        model = resnet18(True)
    if load_local_weight is not None:
        state_dict = torch.load(load_local_weight)
        model.load_state_dict(state_dict)
    return model


def get_resnet34(load_official_weight, load_local_weight=None):
    model = resnet34()
    if load_official_weight:
        model = resnet34(True)
    if load_local_weight is not None:
        state_dict = torch.load(load_local_weight)
        model.load_state_dict(state_dict)
    return model


def get_resnet50(load_official_weight, load_local_weight=None):
    model = resnet50()
    if load_official_weight:
        model = resnet50(True)
    if load_local_weight is not None:
        state_dict = torch.load(load_local_weight)
        model.load_state_dict(state_dict)
    return model


def get_resnet101(load_official_weight, load_local_weight=None):
    model = resnet101()
    if load_official_weight:
        model = resnet101(True)
    if load_local_weight is not None:
        state_dict = torch.load(load_local_weight)
        model.load_state_dict(state_dict)
    return model


def get_resnet152(load_official_weight, load_local_weight=None):
    model = resnet152()
    if load_official_weight:
        model = resnet152(True)
    if load_local_weight is not None:
        state_dict = torch.load(load_local_weight)
        model.load_state_dict(state_dict)
    return model


def change_fc(model, model_name, num_class):
    # 改全连接层
    if model_name == "efficientnet":
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, num_class)
    elif model_name == "resnet":
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_class)
    return model


def load_my_weight(model, model_name, num_class, my_weight_src):
    model = change_fc(model, model_name, num_class)
    state_dict = torch.load(my_weight_src)
    model.load_state_dict(state_dict)
    return model


if __name__ == '__main__':
    net = get_efficientnetb0(False)
    print(net)
