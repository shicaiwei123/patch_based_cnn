import torchvision.transforms as ts

import torch.optim as optim
import os
import numpy as np
from argparse import ArgumentParser

# GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 训练参数
'''
每次需要修改的是--model_name,--log_name,其他如果不是需要调整,可以使用默认参数
retaining 用于解决可能由于意外或者等等情况,训练中途停止了的情况,可以继续重新训练,因为模型参数的保存是保留了训练状态的.
训练状态和模型参数一起保存在模型文件夹里面的,所以只支持对单个模型的训练重复进行
'''
parser = ArgumentParser(description='Pytorch mnist example')

parser.add_argument('--train_epoch', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lrcos', type=bool, default=False, help='using cosine learning rate decay or not ')
parser.add_argument('--lr_warmup', type=bool, default=False)
parser.add_argument('--mixup', type=bool, default=False, help='using mixup or not')
parser.add_argument('--mixup_alpha', type=float, default=0.2)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.99)
parser.add_argument('--class_num', type=int, default=2)
parser.add_argument('--retrain', type=bool, default=False, help='Separate training for the same training process')
parser.add_argument('--log_interval', type=int, default=10, help='How many batches to print the output once')
parser.add_argument('--save_interval', type=int, default=10, help='How many batches to save the model once')
parser.add_argument('--model_root', type=str, default='../output/models')
parser.add_argument('--model_name', type=str, default='mnist_cnn_best.pt')
parser.add_argument('--log_root', type=str, default='../output/logs')
parser.add_argument('--log_name', type=str, default='.csv')

args = parser.parse_args()

args.mixup = False
