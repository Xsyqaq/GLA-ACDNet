import torch
import cv2
import numpy as np
import random

class MaskToTensor(object):
    def __call__(self, img):
        #用于将 NumPy 数组转换为 PyTorch 张量，并将其数据类型转换为长整型（long）
        return torch.from_numpy(img).long()

def get_params(angle_scope=45, scale_scope=0.2, shift_scope=32):
    # 2023年4月23日 random.uniform生成范围内随机小数
    angle = random.uniform(-angle_scope, angle_scope)
    scale = random.uniform(1-scale_scope, 1+scale_scope)
    # 2023年4月23日 random.randint生成范围内随机整数
    shift_x, shift_y = random.randint(-shift_scope, shift_scope), random.randint(-shift_scope, shift_scope)
    return angle, scale, (shift_x, shift_y)

def affine_transform(img, angle, scale, shift, w, h):
    # Rotation
    # 2023年4月23日 cv2.getRotationMatrix2D(旋转中心，旋转角度，旋转后缩放比例)
    MR = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    # 2023年4月23日 得到矩阵后得用图像的仿射变换函数cv2.warpAffine才可以进行最终图像的变化
    img = cv2.warpAffine(img, MR, (w, h))
    # Translation
    # 2023年4月23日 shift起着平移的作用，向左平移shift[0]，向上平移shift[1]
    MT = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    img = cv2.warpAffine(img, MT, (w, h))
    return img

def convert_from_color_annotation(arr_3d):
    arr_3d = arr_3d.astype('uint8')
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    palette = {(255, 255, 255): 0, (0, 0, 255): 255}
    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i
    return arr_2d