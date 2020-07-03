import cv2
import pwd
import os
from skimage.feature import local_binary_pattern
from sklearn.externals import joblib

os.seteuid(pwd.getpwnam(os.getlogin()).pw_uid)

import numpy as np
from scipy.fftpack import fft
import math

import scipy.io as sio
import datetime
import dlib
import copy
from matplotlib import pyplot as plt


class c_face_detection():
    """
    人脸检测类，检测人脸，人脸图像预处理
    """

    def __init__(self):
        """
        构造函数，初始化参数
        """
        self.picture = None  # 缓存有人脸的照片
        self.a_face_detect = None  # 缓存人脸区域内容
        self.a_face_rect = [0, 0, 0, 0]  # 缓存 人脸区域坐标
        self.a_landmark = None  # 缓存特征点
        self.a_face_profile = None
        PREDICTOR_PATH = "/home/shicaiwei/information_form_idcard/face_detection/utils/shape_predictor_68_face_landmarks.dat"
        self.a_detector = dlib.get_frontal_face_detector()
        self.a_predictor = dlib.shape_predictor(PREDICTOR_PATH)
        self.a_face_cascade = cv2.CascadeClassifier(
            '/home/shicaiwei/information_form_idcard/face_detection/utils/haarcascade_frontalface_default.xml')
        self.face_detection_number = 0

    def face_detection(self, img, display=False):
        '''
        输入一张RGB照片，利用opencv的人脸检测器，粗略检测人脸，并且返回人脸图像和人脸位置
         self.a_face_rect: 只可能是[],或者face max
        :return:
        '''

        # 预处理
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 人脸提取
        face_cascade = self.a_face_cascade
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5))

        # 分析
        if len(faces) == 0:
            self.a_face_rect = faces
            return faces
        else:

            # 求最大人脸
            face_max = [0, 0, 0, 0]
            for i in range(len(faces)):
                face = faces[i]
                if face[2] > face_max[2]:
                    face_max = face

            # 人脸截取
            left = face_max[0]
            top = face_max[1]
            right = left + face_max[2]
            bottom = top + face_max[2]
            face_roi = img[top:bottom, left:right]
            self.a_face_detect = face_roi
            self.a_face_rect = face_max

            # 显示
            if display:
                cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
                cv2.imshow("face_roi", self.a_face_detect)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("face_roi display")

            return [face_max]


    def landmark_detection(self, img_rgb, face_rect, display=False):
        '''
        输入包含人脸的RGB照片和人脸区域坐标（opencv），返回特征点
        关键点以坐标的list的方式返回
        display: 是否要显示特诊点
        :return:
        '''

        faces = face_rect
        # 类型转变，opencv_to_dlib
        coordinate = faces
        x1 = coordinate[0]
        y1 = coordinate[1]
        x2 = x1 + coordinate[2]
        y2 = y1 + coordinate[3]
        rect = dlib.rectangle(x1, y1, x2, y2)

        img_key = img_rgb.copy()
        predictor = self.a_predictor
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        points_keys = []

        # 特征点检测,只取第一个,也就是最大的一个
        landmarks = np.matrix([[p.x, p.y] for p in predictor(gray, rect).parts()])

        # 特征点提取,标注
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            points_keys.append([point[0, 0], point[0, 1]])
            cv2.circle(img_key, pos, 2, (255, 0, 0), -1)

        if display:
            cv2.imshow("landmark", img_key)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("landmark display")
        self.a_landmark = points_keys
        return points_keys

    def face_select(self, img_rgb, lamdmarks, display='false'):
        '''
        输入包含人脸的图像和特征点，根据需要选择自己需要的区域，返回
        该函数会根据不同的需要，往往会进行重构
        :param img_rgb:
        :param lamdmarks:
        :return:
        '''

        key = lamdmarks
        left = key[3][0]  # 第一个特征点的纵坐标,也就是宽的方向
        right = key[13][0]  # 第17个特征点的纵坐标,也就是宽方形
        top = key[20][1] - 10  # 第21个点的横坐标,也即是高的起始
        bottom = key[8][1] - 20  # 第9个点的横坐标
        img_roi = img_rgb[top:bottom, left:right]

        # 显示
        if display == 'true':
            cv2.rectangle(img_rgb, (left, top), (right, bottom), (0, 255, 0), 1)
            cv2.imshow("face_roi", img_rgb)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("face_roi display")

        return img_roi

    def color_judge(self, img_rgb):
        '''
        判断图像是不是彩色图片，不死返回False，是返回True
        原理是灰度图片的三通道相差不大
        :return:
        '''
        img = cv2.resize(img_rgb, (300, 300))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray_mean = np.mean(img_gray)
        R = np.float32(img[:, :, 0])
        B = np.float32(img[:, :, 1])
        G = np.float32(img[:, :, 2])
        RG = np.abs(R - G)
        RB = np.abs(R - B)

        zero_mask = RG == 0
        RG_zero = RG[zero_mask]
        RG_zero_num = RG_zero.size

        zero_mask = RB == 0
        RB_zero = RB[zero_mask]
        RB_zero_num = RB_zero.size

        # 超过阈值则表明是黑白照片，直接判为欺骗
        if RB_zero_num > 2000 and RG_zero_num > 2000:
            print("灰色图像")
            return False

        return True

    def face_detection_video(self, cap):
        """
        从摄像机中实时拍摄，检测人脸，返回拍摄有人脸的图片
        :return:
        """
        cap = cap

        while (1):
            # get a frame
            ret, frame = cap.read()
            img = cv2.resize(frame, (640, 480))

            # 人脸识别
            face_rect = self.face_detection(img, display='false')
            if len(face_rect) == 0:
                continue
            else:
                face_rect = face_rect[0]

            # 特征点提取
            landmarks = self.landmark_detection(frame, face_rect, display='false')

            # 人脸选择
            img_roi = self.face_select(frame, landmarks, display='false')

            return img_roi
            # release camera
            cap.release()
            cv2.destroyAllWindows()

    def face_detection_picture(self, p_img, model='process'):
        '''
        从照片中检测人脸，并且根据颜色分布和大小来排除一些欺骗照片、
        '''
        # 全景图片人脸检测
        # 初始化
        begin = datetime.datetime.now()
        frame = p_img
        image_shape = frame.shape

        img = frame
        kernel_size = (5, 5)
        if image_shape[0] > 2000:
            # sigma = 0.5
            # img_process = cv2.GaussianBlur(img, kernel_size, sigma)
            if image_shape[0] >= image_shape[1]:
                img = cv2.resize(frame, (480, 640))
            else:
                img = cv2.resize(frame, (640, 480))

        # 黑白和彩印区别
        color_result = self.color_judge(img)

        # 预处理和实际使用的时候,这个边界条件设置的不一样,是为了更好的限制输入
        # 同时也为了预处理的时候不至于丢掉太多图片
        if model == 'process':
            threshold = 130
        elif model == 'test':
            threshold = 120
            if color_result is False:
                return None
        else:
            threshold = 70

        #  人脸识别
        face_rect = self.face_detection(img, display='false')
        if len(face_rect) == 0:
            return None
        else:
            face_rect = face_rect[0]

        # 特征点提取
        landmarks = self.landmark_detection(img, face_rect, display='false')

        # 求端点,分割处人脸
        key = landmarks
        left = key[3][0]  # 第一个特征点的纵坐标,也就是宽的方向
        right = key[13][0]  # 第17个特征点的纵坐标,也就是宽方形
        distance = key[27][1] - key[21][1]  # 使用相对间距而不是绝对间距来删选
        top = key[19][1] - distance  # 第21个点的横坐标,也即是高的起始
        bottom = int((key[7][1] + key[9][1]) / 2)  # 第8个点的横坐标
        img_roi = img[top:bottom, left:right]

        # # 删除过于小尺寸的,有一个边小于70,就删除
        if bottom - top < threshold or right - left < threshold:
            return None
        #
        # # 删除过大的
        # if bottom - top > 330 or right - left > 280:
        #     return None

        end = datetime.datetime.now()
        time_sub = end - begin
        # print("人脸截取时间:",time_sub.total_seconds())

        # 显示截取之后的人脸
        # cv2.imshow("face",img_roi)
        # cv2.waitKey(0)

        return img_roi

    def get_face_picture(self):
        return self.a_face_picture

    def face_posture(self, img_rgb, display=False):
        '''
        求人脸的角度倾斜，水平倾斜，前后倾斜
        :param img_rgb:
        :param display:
        :return:
        '''
        face = self.a_face_rect
        if len(face) == 0:
            return None, None


        # 特征点提取
        landmarks = self.landmark_detection(img_rgb, face, display=False)
        self.a_landmark = landmarks

        if display:
            cv2.imshow("frame", img_rgb)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("face_roi display")

        key = landmarks

        # 求两只眼睛的中心坐标
        left_eye = [36, 37, 38, 39, 40, 41]
        right_eye = [42, 43, 44, 45, 46, 47]
        left_eye_x = 0
        left_eye_y = 0
        right_eye_x = 0
        right_eye_y = 0
        for i in left_eye:
            left_eye_x += key[i][0] * 1 / 6
            left_eye_y += key[i][1] * 1 / 6
        for i in right_eye:
            right_eye_x += key[i][0] * 1 / 6
            right_eye_y += key[i][1] * 1 / 6

        dx = (right_eye_x - left_eye_x)  # note: right - right
        dy = (right_eye_y - left_eye_y)
        # print("dx",dx,"dy",dy)

        # 求人脸两边特征点到中间的距离
        left_to_center = key[29][0] - key[2][0]
        right_to_center = key[14][0] - key[29][0]
        distance = left_to_center - right_to_center

        # 阈值40，角度5

        angle = math.atan2(dy, dx) * 180. / math.pi  # 计算角度

        return angle, distance

    def face_alignment(self, cap):
        # 初始化
        behin = datetime.datetime.now()
        predictor = self.a_predictor
        face_cascade = self.a_face_cascade
        faces_aligned = []

        while 1:

            ret, frame = cap.read()
            if frame is None:
                continue

            picture = frame

            angle, distance = self.face_posture(picture)
            print("angle", angle, "distance", distance)

            # 矫正
            # RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)  # 计算仿射矩阵
            # RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1]))  # 进行放射变换，即旋转
            # faces_aligned.append(RotImg)
            # 计算时间
            end = datetime.datetime.now()
            time_sub = end - behin
            # print("人脸对齐时间:", time_sub.total_seconds())

            # 返回
            # return faces_aligned


if __name__ == '__main__':
    # 人脸检测
    cap = cv2.VideoCapture(1)
    # 人脸检测
    o_face_detection = c_face_detection()
    while 1:
        # ret, frame = cap.read()
        # img = frame

        # 预处理
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        o_face_detection.face_alignment(cap)
        # face_picture = o_face_detection.face_detetion_picture(img)
