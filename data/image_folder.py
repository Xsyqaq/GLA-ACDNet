"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    #2023年4月21日 判断filename的后缀结尾是不是IMG_EXTENSIONS中的一种
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    # 2023年4月21日 判断dir是不是目录，不是就报错%s is not a valid directory
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, dir_names, file_names in sorted(os.walk(dir)):
        # 2023年4月21日 os.walk沿着路径逐层输出路径，dir，files
        for fname in file_names:
            if is_image_file(fname):
                # 2023年4月21日 判断图片类型是否是IMG_EXTENSIONS中的一种
                path = os.path.join(root, fname)
                images.append(path)
    # 2023年4月21日 images列表装了每张图片的绝对路径
    return images[:min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False, loader=default_loader):
        # 2023年4月21日 imgs列表装了每张图片的绝对路径
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        # 2023年4月21日 默认Image.open方式打开转换RGB
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
