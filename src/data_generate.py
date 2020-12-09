import sys

sys.path.append('..')

from lib.processing_utils import get_file_list, FaceDection
import torchvision.transforms as ts
import cv2
from PIL import Image
import numpy as np
import os
import torch.functional as F


class RandomCrop(object):

    def __init__(self, size, seed):
        self.seed = seed
        self.size = (int(size), int(size))

    def get_params(self, img, output_size):
        img = np.array(img)
        img_shape = img.shape
        w = img_shape[0]
        h = img_shape[1]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        np.random.seed(self.seed)
        i = np.random.randint(0, h - th)
        j = np.random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.size)
        img = np.array(img)
        img_new = img[j:j + h, i:i + w]
        try:
            img_new = Image.fromarray(img_new.astype('uint8')).convert('RGB')
        except Exception as e:
            print("Image.fromarray(img.astype('uint8')).convert('RGB')")
            i, j, h, w = self.get_params(img, self.size)
        return img_new


def living_face_patch_generate(living_face, save_dir, patch_size=96, display=False):
    # initial

    data_len = 8
    seed_arr = np.arange(data_len)
    img_saved_list = get_file_list(save_dir)
    img_patch_num = len(img_saved_list)
    img_Image = Image.fromarray(cv2.cvtColor(living_face, cv2.COLOR_BGR2RGB))

    for i in range(data_len):
        img_transform = RandomCrop(size=patch_size, seed=seed_arr[i])
        try:
            img_patch = img_transform(img_Image)
            if display:
                img_patch_opencv = cv2.cvtColor(np.array(img_patch), cv2.COLOR_RGB2BGR)
                cv2.imshow("patch", img_patch_opencv)
                cv2.waitKey(0)

            img_patch_num += 1
            img_patch_name = "patch" + str(img_patch_num) + ".bmp"
            patch_save_path = save_dir + "/" + img_patch_name

            img_patch_opencv = cv2.cvtColor(np.array(img_patch), cv2.COLOR_RGB2BGR)
            cv2.imwrite(patch_save_path, img_patch_opencv)
        except Exception as e:
            print(e)


def spoofing_face_patch_generate(spoofing_face, save_dir, patch_size=96, display=False):
    # nitial
    img_transform = ts.Compose([ts.RandomCrop(patch_size)])
    img_saved_list = get_file_list(save_dir)
    img_patch_num = len(img_saved_list)
    img_Image = Image.fromarray(cv2.cvtColor(spoofing_face, cv2.COLOR_BGR2RGB))
    data_len = 8
    seed_arr = np.arange(data_len)

    for i in range(data_len):
        img_transform = RandomCrop(size=patch_size, seed=seed_arr[i])
        try:
            img_patch = img_transform(img_Image)
            if display:
                img_patch_opencv = cv2.cvtColor(np.array(img_patch), cv2.COLOR_RGB2BGR)
                cv2.imshow("patch", img_patch_opencv)
                cv2.waitKey(0)

            img_patch_num += 1
            img_patch_name = "patch" + str(img_patch_num) + ".bmp"
            patch_save_path = save_dir + "/" + img_patch_name

            img_patch_opencv = cv2.cvtColor(np.array(img_patch), cv2.COLOR_RGB2BGR)
            cv2.imwrite(patch_save_path, img_patch_opencv)
        except Exception as e:
            print(e)


def patch_generate(face_dir, patch_save_dir, sample_interal=1):
    '''
    将人脸裁剪成相同大小的patch
    :param face_dir:
    :param patch_save_dir:
    :sample_interal 对抗正负样本不均衡
    :return:
    '''
    count = 1
    for root, dirs, files in os.walk(face_dir):

        # 当子目录为空的时候，root就是不包含子文件夹的文件夹
        if dirs == []:
            files = sorted(files)
            for file_name in files:
                file_path = os.path.join(root, file_name)
                img = cv2.imread(file_path)
                if img is None:
                    print("if img is None:")
                    continue

                # 获取存储路径
                face_dir_split = face_dir.split('/')
                file_path_split = file_path.split('/')
                file_path_split.pop()  # 去掉文件名
                sub_split = [item for item in file_path_split if item not in face_dir_split]
                save_dir = patch_save_dir
                for item in sub_split:
                    save_dir = os.path.join(save_dir, item)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                # 截取patch
                count += 1

                # index
                if file_path_split[-1] == 'spoofing':
                    if count % sample_interal != 0:
                        print(count)
                        continue
                    spoofing_face_patch_generate(spoofing_face=img, save_dir=save_dir)
                else:
                    living_face_patch_generate(living_face=img, save_dir=save_dir)
                    print(count)



if __name__ == '__main__':
    face_dir = "/home/bbb//shicaiwei/data/liveness_data/cross_replayed_face_normal"
    patch_save_dir = "/home/bbb/shicaiwei/data/liveness_data/cross_replayed_patch_normal"
    patch_generate(face_dir=face_dir, patch_save_dir=patch_save_dir, sample_interal=1)
