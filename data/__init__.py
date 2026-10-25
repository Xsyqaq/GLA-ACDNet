"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset

#根据dataset_name返回data文件夹下对应的模块
def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    # 2023年5月5日 data.deepcrack_dataset
    dataset_filename = "data." + dataset_name + "_dataset"    #data.deepcrack_dataset
    #  2023年5月5日
    #  < module'data.deepcrack_dataset'from'D:\\PythonProject\\DeepSegmentor-master\\data\\deepcrack_dataset.py' >
    datasetlib = importlib.import_module(dataset_filename)   # 导入data文件夹下deepcrack_dataset.py模块

    #  2023年5月5日 不同的数据集需要不同的处理方法，这里用到了deepcrack和roadnet，如果你要用crack500，就要自己
    # 在data文件下编一个对应py，否则dataset=none就会返回一个44行的错误提示
    dataset = None
    # 2023年5月5日 deepcrackdataset
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'  #deepcrackdataset
    for name, cls in datasetlib.__dict__.items():

        #dataset=(data文件下deepcrack_dataset.py的class DeepCrackDataset)
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    # 2022年5月5日 输出<function BaseDataset.modify_commandline_options at 0x000001464B078AF8>，暂且认为未使用
    dataset_class = find_dataset_using_name(dataset_name)    #data文件下deepcrack_dataset.py的class DeepCrackDataset
    return dataset_class.modify_commandline_options    #返回DeepCrackDataset继承的basedataset的静态方法


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)   #返回deepcrack_dataset.py中class DeepCrackDataset
        self.dataset = dataset_class(opt)    # 读取了数据集并将图片进行了处理
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)
    # 该方法从 self.dataloader 中批量读取数据，并在达到指定的最大数据集大小时停止迭代。
    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            # 2023年4月24日 enumerate返回值有两个。一个是序号，一个是数据（包含训练数据和标签）
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            # 2023年4月24日 迭代返回data
            yield data
