import os.path
import random
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from data.utils import MaskToTensor, get_params, affine_transform

class DeepCrackDataset(BaseDataset):
    """A dataset class for crack dataset."""

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.img_paths = make_dataset(os.path.join(opt.dataroot, '{}_img'.format(opt.phase)))  #路径如./datasets/DeepCrack/train_img
        self.lab_dir = os.path.join(opt.dataroot, '{}_lab'.format(opt.phase))  #路径如./datasets/DeepCrack/train_lab
        # 图片预处理
        # transforms.Normalize 用于图像的逐通道标准化。mean: 每个通道的均值。std: 每个通道的标准差。
        self.img_transforms = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5),
                                                                       (0.5, 0.5, 0.5))])
        # 用于将 NumPy 数组转换为 PyTorch 张量，并将其数据类型转换为长整型（long）
        self.lab_transform = MaskToTensor()
        
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            image (tensor) - - an image
            label (tensor) - - its corresponding segmentation
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        img_path = self.img_paths[index]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # 读取图像，但不进行任何颜色转换
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lab_path = os.path.join(self.lab_dir, os.path.basename(img_path).split('.')[0]+'.png')  #标签路径,根据标签实际情况改为对应的后缀格式
        lab = cv2.imread(lab_path, cv2.IMREAD_UNCHANGED)
        if len(lab.shape) == 3:
            lab = cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)
        
        # adjust the image size
        # load_width和load_height由命令注入分别为256，256
        w, h = self.opt.load_width, self.opt.load_height
        if w > 0 or h > 0:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            lab = cv2.resize(lab, (w, h), interpolation=cv2.INTER_CUBIC)

        # binarize segmentation
        # 根据设定的阈值处理标签图像的灰度值，>127,取255，其他取0(_指阈值127，lab指返回的图像)
        _, lab = cv2.threshold(lab, 127, 255, cv2.THRESH_BINARY)

        # apply flip
        # no_flip由命令注入0，
        if (not self.opt.no_flip) and random.random() > 0.5:
            # >0.5图像左右水平翻转，<0.5图像上下垂直翻转
            if random.random() > 0.5:
                img = np.fliplr(img)
                lab = np.fliplr(lab)
            else:
                img = np.flipud(img)
                lab = np.flipud(lab)

        # apply affine transform
        if self.opt.use_augment:
            if random.random() > 0.5:
                # data.utils.get_params(),得到旋转角度，旋转后缩放比例，平移距离
                angle, scale, shift = get_params()
                img = affine_transform(img, angle, scale, shift, w, h)
                lab = affine_transform(lab, angle, scale, shift, w, h)

        _, lab = cv2.threshold(lab, 127, 1, cv2.THRESH_BINARY)

        # apply the transform to both A and B
        #  实现array到image的转换,np.asarray(image)PIL image转换成array
        img = self.img_transforms(Image.fromarray(img.copy()))
        lab = self.lab_transform(lab.copy()).unsqueeze(0)

        return {'image': img, 'label': lab, 'A_paths': img_path, 'B_paths': lab_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_paths)
