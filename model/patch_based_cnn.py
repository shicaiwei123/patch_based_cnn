
import torchvision.transforms as ts
import numpy as np
import os
import torch.nn as nn
import torch
from torchvision.datasets import ImageFolder
import torch.nn.init as init
import torch.optim as optim


class net_baesd_patch(nn.Module):
    def __init__(self, args):
        super(net_baesd_patch, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 50, 5, 1, padding=2),
            nn.BatchNorm2d(50, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(50, 100, 3, 1, padding=1),
            nn.BatchNorm2d(100, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(100, 150, 3, 1, padding=1),
            nn.BatchNorm2d(150, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(150, 200, 3, 1, padding=1),
            nn.BatchNorm2d(200, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(200, 250, 3, 1, padding=1),
            nn.BatchNorm2d(250, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(250, 1000, 3, 1),
            nn.BatchNorm2d(1000, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5, inplace=False),

        )

        self.classifier = nn.Sequential(
            nn.Linear(1000, 400),
            nn.BatchNorm1d(400, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Linear(400, args.class_num),

        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


def net_init(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            print(m.bias)
            init.normal(m.weight)
            if m.bias is not None:
                init.constant(m.bias, 0)
        if isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)

        if isinstance(m, nn.Conv2d):
            init.kaiming_uniform(m.weight)


class linear_test(nn.Module):
    def __init__(self):
        super(linear_test, self).__init__()
        self.fc1 = nn.Linear(20, 2, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        return x


def get_mean_std(dataset, ratio=1):
    """Get mean and std by sample ratio
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(len(dataset) * ratio),
                                             shuffle=True, num_workers=10)
    train = iter(dataloader).next()[0]  # 一个batch的数据
    mean = np.mean(train.numpy(), axis=(0, 2, 3))
    std = np.std(train.numpy(), axis=(0, 2, 3))
    return mean, std


class rgb2ycrcb(object):
    '''
    自定义transform处理,将rgb图像转ycrcb
    :param object:
    :return:
    '''

    def __call__(self, img):
        img_new = img.convert("YCbCr")
        return img_new


def my_data_loader(train=True, batch_size=100):
    """
    :param train: train or test fold?
    :param batch_size: batch size, int
    :return: data loader

    """
    print("load data...")
    train_transform = ts.Compose(
        [rgb2ycrcb(),
         ts.ToTensor(),
         ts.Normalize(mean=(0.53, 0.43, 0.59,), std=(0.18, 0.03, 0.04))])

    test_transform = ts.Compose(
        [rgb2ycrcb(),
         ts.ToTensor(),
          ts.Normalize(mean=(0.53, 0.43, 0.59,), std=(0.18, 0.03, 0.04))])

    train_data_root = '../data/train'
    test_data_root = '../data/test'
    train_data_set = ImageFolder(train_data_root, transform=train_transform)
    test_data_set = ImageFolder(test_data_root, transform=test_transform)

    #train_mean, train_std = get_mean_std(train_data_set)
    #test_mean, test_std = get_mean_std(test_data_set)
    #print("train_mean", train_mean, "train_std", train_std)
    #print("test_mean", test_mean, "test_std", test_std)

    if train:
        loader = torch.utils.data.DataLoader(train_data_set, batch_size=batch_size,
                                             shuffle=True, num_workers=4)
    else:

        loader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size,
                                             shuffle=False, num_workers=4)
    print("load finish")
    return loader
