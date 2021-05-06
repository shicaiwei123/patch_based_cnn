import torchvision.transforms as ts
import numpy as np
import os
import torch.nn as nn
import torch
from PIL import Image
import cv2
from torchvision.datasets import ImageFolder
import torch.nn.init as init
import torch.optim as optim
from lib.processing_utils import get_mean_std
from lib.img_dataset import ImgPixelDataset


class net_baesd_depth(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(128, affine=True, track_running_stats=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.pool2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 160, 3, 1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(160, affine=True, track_running_stats=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.upsample1 = nn.Sequential(

            nn.Conv2d(160, 128, 3, 1, 1),
            # 'there are some errors in the paper, stride should be 1'
            nn.ConvTranspose2d(128, 128, 6, 1, 0),
        )

        self.upsample2 = nn.Sequential(

            nn.Conv2d(128, 128, 3, 1, 1),
            # 'there are some errors in the paper, stride should be 1'
            nn.ConvTranspose2d(128, 128, 6, 1, 0),
        )

        self.upsample3 = nn.Sequential(

            nn.Conv2d(128, 160, 3, 1, 1),
            # 'there are some errors in the paper, stride should be 1'
            nn.ConvTranspose2d(160, 160, 6, 1, 0),
        )

        self.upsample4 = nn.Sequential(

            nn.Conv2d(160, 320, 3, 1, 1),
            # 'there are some errors in the paper, stride should be 1'
            nn.ConvTranspose2d(320, 320, 6, 1, 0),
        )

        self.conv = nn.Conv2d(320, 1, 3, 1, 1)

    def forward(self, x):
        x = self.pool1(x)
        x = self.pool2(x)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        x=self.conv(x)
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


class rgb2ycrcb(object):
    '''
    自定义transform处理,将rgb图像转ycrcb
    :param object:
    :return:
    '''

    def __call__(self, img):
        img_new = img.convert("YCbCr")
        return img_new


depth_train_transform = ts.Compose(
    [
        ts.Resize((128, 128)),
        rgb2ycrcb(),
        ts.ToTensor(),
        ts.Normalize(mean=(0.56, 0.45, 0.58,), std=(0.18, 0.04, 0.04))  # cross photo
        # ts.Normalize(mean=(0.52, 0.45, 0.57,), std=(0.17, 0.04, 0.04))  # cross replayed
        # ts.Normalize(mean=(0.44, 0.47, 0.55,), std=(0.15, 0.02, 0.02))  # fasd
    ])

depth_test_transform = ts.Compose(
    [
        ts.Resize((128, 128)),
        rgb2ycrcb(),
        ts.ToTensor(),
        ts.Normalize(mean=(0.56, 0.45, 0.58,), std=(0.18, 0.04, 0.04))  # cross photo
        # ts.Normalize(mean=(0.560, 0.46, 0.57,), std=(0.16, 0.04, 0.04))  # cross replayed
        # ts.Normalize(mean=(0.44, 0.47, 0.55,), std=(0.15, 0.02, 0.02))
    ])


def depth_data_loader(args, train=True):
    """de
    :param train: train or test fold?
    :param batch_size: batch size, int
    :return: data loader

    """

    print("load data")

    living_train_dir = os.path.join(args.train_dir, 'living')
    living_test_dir = os.path.join(args.test_dir, 'living')
    spoofing_train_dir = os.path.join(args.train_dir, 'spoofing')
    spoofing_test_dir = os.path.join(args.test_dir, 'spoofing')

    train_data_set = ImgPixelDataset(living_dir=living_train_dir, spoofing_dir=spoofing_train_dir,
                                     data_transform=depth_train_transform, pixel_size=52)
    test_data_set = ImgPixelDataset(living_dir=living_test_dir, spoofing_dir=spoofing_test_dir,
                                    data_transform=depth_test_transform, pixel_size=52)

    # train_mean, train_std = get_mean_std(train_data_set)
    # test_mean, test_std = get_mean_std(test_data_set)
    # print("train_mean", train_mean, "train_std", train_std)
    # print("test_mean", test_mean, "test_std", test_std)

    if train:
        loader = torch.utils.data.DataLoader(train_data_set, batch_size=args.batch_size,
                                             shuffle=True, num_workers=4)
    else:

        loader = torch.utils.data.DataLoader(test_data_set, batch_size=args.batch_size,
                                             shuffle=False, num_workers=4)
    return loader
