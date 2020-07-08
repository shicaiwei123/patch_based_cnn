from patch_based_cnn.src.FaceDetection import c_face_detection
from patch_based_cnn.utils import get_file_list
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


def living_face_patch_generate(living_face, save_path, patch_size=96, display=False):
    # initial

    data_len = 8
    seed_arr = np.arange(data_len)
    img_saved_list = get_file_list(save_path)
    img_patch_num = len(img_saved_list)
    img_Image = Image.fromarray(cv2.cvtColor(living_face, cv2.COLOR_BGR2RGB))

    for i in range(data_len):
        img_transform = RandomCrop(size=patch_size, seed=seed_arr[i])
        img_patch = img_transform(img_Image)
        if display:
            img_patch_opencv = cv2.cvtColor(np.array(img_patch), cv2.COLOR_RGB2BGR)
            cv2.imshow("patch", img_patch_opencv)
            cv2.waitKey(0)

        img_patch_num += 1
        img_patch_name = "patch" + str(img_patch_num) + ".bmp"
        patch_save_path = save_path + "/" + img_patch_name

        img_patch_opencv = cv2.cvtColor(np.array(img_patch), cv2.COLOR_RGB2BGR)
        cv2.imwrite(patch_save_path, img_patch_opencv)


def spoofing_face_patch_generate(spoofing_face, save_path, patch_size=96, display=False):
    # nitial
    img_transform = ts.Compose([ts.RandomCrop(patch_size)])
    img_saved_list = get_file_list(save_path)
    img_patch_num = len(img_saved_list)
    img_Image = Image.fromarray(cv2.cvtColor(spoofing_face, cv2.COLOR_BGR2RGB))
    data_len = 8
    seed_arr = np.arange(data_len)

    for i in range(data_len):
        img_transform = RandomCrop(size=patch_size, seed=seed_arr[i])
        img_patch = img_transform(img_Image)
        if display:
            img_patch_opencv = cv2.cvtColor(np.array(img_patch), cv2.COLOR_RGB2BGR)
            cv2.imshow("patch", img_patch_opencv)
            cv2.waitKey(0)

        img_patch_num += 1
        img_patch_name = "patch" + str(img_patch_num) + ".bmp"
        patch_save_path = save_path + "/" + img_patch_name

        img_patch_opencv = cv2.cvtColor(np.array(img_patch), cv2.COLOR_RGB2BGR)
        cv2.imwrite(patch_save_path, img_patch_opencv)


def patch_generate(img_path, train, face_img_path=None, is_Liveing=True, display=False):
    '''
    根据输入的picture, divide they into different pacth
    请将测试数据和训练数据分别输入
    :param img_path:  path of imgs before face detection
    :param face_img_path: path of img of face region
    :param is_Liveing: if it's true,the img is came from living man,otherwise,it's spoofing img
    :return:
    '''
    # intial
    o_face_detection = c_face_detection()
    face_region = None
    if train:
        living_save_path = "/home/shicaiwei/information_form_idcard/face_detection/reproduce/patch_based_cnn/data/train/true"
        spoofing_save_path = "/home/shicaiwei/information_form_idcard/face_detection/reproduce/patch_based_cnn/data/train/false"
    else:
        living_save_path = "/home/shicaiwei/information_form_idcard/face_detection/reproduce/patch_based_cnn/data/test/true"
        spoofing_save_path = "/home/shicaiwei/information_form_idcard/face_detection/reproduce/patch_based_cnn/data/test/false"

    if not os.path.exists(living_save_path):
        os.makedirs(living_save_path)
    if not os.path.exists(spoofing_save_path):
        os.makedirs(spoofing_save_path)

    # 获取人脸照片
    if face_img_path is None:
        img_file_list = get_file_list(img_path)
        for img_path in img_file_list:
            img = cv2.imread(img_path)

            if img is None:
                print("img is None")
                continue
            # img= Image.open(img_path)
            img = cv2.resize(img, (480, 640))

            face_rects = o_face_detection.face_detection(img)
            if len(face_rects) == 0:
                print("face_rect is 0")
                continue

            face_rect = face_rects[0]

            landmarks = o_face_detection.landmark_detection(img, face_rect)
            face_region = o_face_detection.face_select(img, landmarks)

            if is_Liveing:
                living_face_patch_generate(face_region, living_save_path, display=display)
            else:
                spoofing_face_patch_generate(face_region, spoofing_save_path, display=display)
    else:
        face_img_file_list = get_file_list(face_img_path)
        for face_img_path in face_img_file_list:
            face_region = cv2.imread(face_img_path)

            if is_Liveing:
                living_face_patch_generate(face_region, living_save_path, display=display)
            else:
                spoofing_face_patch_generate(face_region, spoofing_save_path, display=display)


if __name__ == '__main__':
    img_full_path = "/home/shicaiwei/data/liveness_data/light_false/test"
    train = False
    patch_generate(img_full_path, train, is_Liveing=False)
