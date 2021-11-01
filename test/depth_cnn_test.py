import sys
import cv2
import numpy as np
from PIL import Image
import torch

sys.path.append('../')

from model.depth_based_cnn import net_baesd_depth, depth_test_transform
from lib.processing_utils import get_file_list, FaceDection
from lib.model_develop_utils import deploy_base
from configuration.config_depth import args
import torchvision.transforms as ts
import datetime


class rgb2ycrcb(object):
    '''
    自定义transform处理,将rgb图像转ycrcb
    :param object:
    :return:
    '''

    def __call__(self, img):
        img_new = img.convert("YCbCr")
        return img_new


def depth_cnn_single(model, face_detector, img, isface):
    '''

    :param model:
    :param face_detector:
    :param img:
    :param isface: the img is face img or not
    :return:
    '''
    if not isface:
        img = cv2.resize(img, (480, 640))
        # 人脸检测
        face_img = face_detector.face_detect(img)

    else:
        face_img = img

    if face_img is None:
        return None

    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img_pil = Image.fromarray(face_img)

    result = deploy_base(model=model, img=face_img_pil, transform=depth_test_transform)

    result_mean = np.mean(result)
    print(result_mean)

    if result_mean < 0.5:
        return 0
    else:
        return 1


def depth_cnn_test(pre_path, test_dir, label, isface):
    '''

    :param :
    :param pre_path: road to pretrain model
    :param test_dir: test img dir
    :param isface: img is face img or not, if not need to detect face
    :return:
    '''
    face_detector = FaceDection(model_name='cv2')
    model = net_baesd_depth()

    state_dict = torch.load(pre_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    time_begin = datetime.datetime.now()
    file_list = get_file_list(test_dir)
    count = 1
    true_num = 1
    for file in file_list:
        img = cv2.imread(file)
        if img is None:
            continue

        result = depth_cnn_single(model=model, face_detector=face_detector, img=img, isface=isface)

        if result is None:
            continue
        if result == label:
            count += 1
            true_num += 1
        else:
            print(file)
            count += 1
    print(count, true_num, true_num / count)

    time_end = datetime.datetime.now()
    time_all = time_end - time_begin
    print("time_all", time_all.total_seconds())


if __name__ == '__main__':
    test_dir = "/home/shicaiwei/data/liveness_data/CASIA-FASD/test/living"
    label = 0
    pre_path = "../output/models/depth_patch.pth"
    isface = True
    depth_cnn_test(pre_path=pre_path, test_dir=test_dir, label=label,
                   isface=isface)
