

import numpy as np
from img_process import get_file_list
import pickle
import cv2
import torch
import torch
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import time
import csv
import os
from torchtoolbox.tools import mixup_criterion, mixup_data
from torchvision.datasets import ImageFolder
import sys
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


def get_file_list(read_path):
    '''
    获取文件夹下图片的地址
    :param read_path:
    :return:
    '''
    path = read_path
    dirs = os.listdir(path)
    floder_len = len(dirs)
    file_name_list = []
    for i in range(floder_len):

        # 设置路径
        floder = dirs[i]
        floder_path = path + "/" + floder

        # 如果路径下是文件，那么就再次读取
        if os.path.isdir(floder_path):
            file_one = os.listdir(floder_path)
            file_len_one = len(file_one)
            for j in range(file_len_one):
                # 读取视频
                floder_path_one = floder_path + "/" + file_one[j]
                if os.path.isdir(floder_path_one):
                    file_two = os.listdir(floder_path_one)
                    file_len_two = len(file_two)
                    for k in range(file_len_two):
                        floder_path_two = floder_path_one + "/" + file_two[k]
                        if os.path.isdir(floder_path_two):
                            file_three = os.listdir(floder_path_two)
                            file_len_three = len(file_three)
                            for m in range(file_len_three):
                                floder_path_three = floder_path_two + "/" + file_three[m]
                                file_name_list.append(floder_path_three)
                        else:
                            file_name_list.append(floder_path_two)

                else:
                    file_name_list.append(floder_path_one)

        # 如果路径下，没有文件夹，直接是文件，就加入进来
        else:
            file_name_list.append(floder_path)

    return file_name_list



def calc_accuracy(model, loader, verbose=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []
    for inputs, labels in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):
        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        with torch.no_grad():
            outputs_batch = model(inputs)
        outputs_full.append(outputs_batch)
        labels_full.append(labels)
    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    return accuracy


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, args, optimizer, multiplier, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = args.total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                # return self.after_scheduler.get_last_lr()
                return [group['lr'] for group in self.optimizer.param_groups]
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


def train_base(model, cost, optimizer, train_loader, test_loader, args):
    '''

    :param model: 要训练的模型
    :param cost: 损失函数
    :param optimizer: 优化器
    :param train_loader:  测试数据装载
    :param test_loader:  训练数据装载
    :param args: 配置参数
    :return:
    '''

    # 打印训练参数
    print(args)

    # 初始化,打开定时器,创建保存位置
    start = time.time()

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.model_name
    log_dir = args.log_root + '/' + args.log_name

    # 保存argv参数
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    # 判断是否使用余弦衰减
    if args.lrcos:
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch, eta_min=0)

        # 是否使用学习率预热
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # 训练初始化
    epoch_num = args.train_epoch  # 训练多少个epoch
    log_interval = args.log_interval  # 每隔多少个batch打印一次状态
    save_interval = args.save_interval  # 每隔多少个epoch 保存一次数据

    batch_num = 0
    train_loss = 0
    log_list = []  # 需要保存的log数据列表

    epoch = 0
    accuracy_best = 0

    # 如果是重新训练的过程,那么就读取之前训练的状态
    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # 训练
    while epoch < epoch_num:  # 以epoch为单位进行循环
        for batch_idx, (data, target) in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            if args.mixup:
                mixup_alpha = args.mixup_alpha
                inputs, labels_a, labels_b, lam = mixup_data(data, target, alpha=mixup_alpha)

            optimizer.zero_grad()  # 优化器梯度初始化为零,不然会累加之前的梯度

            output = model(data)  # 把数据输入网络并得到输出，即进行前向传播

            if args.mixup:
                loss = mixup_criterion(cost, output, labels_a, labels_b, lam)
            else:
                loss = cost(output, target)

            train_loss += loss.item()
            loss.backward()  # 反向传播求出输出到每一个节点的梯度
            optimizer.step()  # 根据输出到每一个节点的梯度,优化更新参数```````````````````````````````````````````````````
            # if batch_idx % log_interval == 0:  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))

        # 测试模型
        accuracy_test = calc_accuracy(model, loader=test_loader)
        if accuracy_test > accuracy_best:
            accuracy_best = accuracy_test
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                        train_loss / len(train_loader),
                                                                                        accuracy_test, accuracy_best))
        train_loss = 0

        if args.lrcos:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # 保存模型和优化器参数
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            torch.save(train_state, models_dir)

        # 保存log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def test_base(model, cost, test_loader):
    '''
    :param model: 要测试的带参数模型
    :param cost: 损失函数
    :param test_loader: 测试集
    :return:
    '''

    model.eval()  # 设置为test模式
    test_loss = 0  # 初始化测试损失值为0
    correct = 0  # 初始化预测正确的数据个数为0
    true_wrong = 0
    true_right = 0
    false_wrong = 0
    false_right = 0

    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        output = model(data)  # 数据是按照batch的形式喂入的,然后是这里的输出是全连接层的输出结果
        test_loss += (
            cost(output, target)).item()  # sum up batch 求loss 把所有loss值进行累加.一般分类器会有softmax,但是这里没有是因为集成在这个损失函数之中了
        pred = output.data.max(1, keepdim=True)[
            1]  # get the index of the max log-probability #输出的结果虽然会过sofamax转成概率值,但是相对打大小是不变的,所以直接输出取最大值对应的索引就是分类结果
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()  # 对预测正确的数据个数进行累加
        compare_result = pred.eq(target.data.view_as(pred)).cpu()  # 判断输出和目标数据是不是一样,然后将比较结果转到cpu上
        # target=target.numpy()
        target = target.cpu()
        compare_result = np.array(compare_result)
        for i in range(len(compare_result)):
            if compare_result[i]:
                if target[i] == 1:
                    true_right += 1
                else:
                    false_right += 1
            else:
                if target[i] == 1:
                    true_wrong += 1
                else:
                    false_wrong += 1

    test_loss /= len(test_loader.dataset)  # 因为把所有loss值进行过累加，所以最后要除以总得数据长度才得平均loss
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
