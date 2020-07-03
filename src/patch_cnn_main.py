import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.utils.data
from tqdm import tqdm
from torchtoolbox.tools import mixup_data, mixup_criterion
import os
import csv
import sys

sys.path.append('../')

from model.patch_based_cnn import net_baesd_patch, my_data_loader
from patch_based_cnn.utils import train_base, test_base, calc_accuracy
from config import args


def patch_cnn_train():
    '''
    基于vgg模型的修改
    re_conv and de_pooling are False means the origin VGG
    re_conv is True means the replace of vgg
    de_pooling is True means the pool doesn't using
    :return:
    '''
    train_loader = my_data_loader(train=True, batch_size=args.batch_size)
    test_loader = my_data_loader(train=False, batch_size=args.batch_size)

    name = 'patch_baed_cnn'
    args.log_name = name + '.csv'
    args.model_name = name + '.pt'

    model = net_baesd_patch(args=args)
    # 如果有GPU
    if torch.cuda.is_available():
        model.cuda()  # 将所有的模型参数移动到GPU上
        print("GPU is using")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    args.retrain = False
    train_base(model=model, cost=criterion, optimizer=optimizer, train_loader=train_loader, test_loader=test_loader,
               args=args)


if __name__ == '__main__':
    # train includes test phase at each epoch
    patch_cnn_train()
