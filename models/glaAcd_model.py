# Author: Yahui Liu <yahui.liu@uintn.it>

import torch
import numpy as np
import itertools

import torch.nn as nn

from .base_model import BaseModel
from .glaAcd_networks import define_deepcrack, BinaryFocalLoss

class glaAcdModel(BaseModel):
    """
    This class implements the DeepCrack model.
    DeepCrack paper: https://www.sciencedirect.com/science/article/pii/S0925231219300566
    """
    # deepcrack模型添加的参数
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options."""
        parser.add_argument('--lambda_side', type=float, default=1.0, help='weight for side output loss')
        parser.add_argument('--lambda_fused', type=float, default=1.0, help='weight for fused loss')
        return parser

    def __init__(self, opt):
        """Initialize the DeepCrack class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>指定要打印的训练损失。训练/测试脚本将调用<BaseModel.get_current_losses>
        self.loss_names = ['side', 'fused', 'total']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>指定你想要保存/显示的图像。训练/测试脚本将调用< basmodel .get_current_visuals>
        self.display_sides = opt.display_sides  #opt.display_sides=1
        self.visual_names = ['image', 'label_viz', 'fused']
        if self.display_sides:
            self.visual_names += ['side1', 'side2', 'side3', 'side4', 'side5']  #设定要可视化输出的层名
        # specify the models you want to save to the disk. 指定要保存到磁盘的模型。
        self.model_names = ['G']

        # define networks
        # opt.input_nc=3， opt.num_classes=1, opt.ngf=64, opt.norm=batch, opt.init_type=xavier,  opt.init_gain=0.02, self.gpu_ids=1
        self.netG = define_deepcrack(opt.input_nc,
                                     opt.num_classes,
                                     opt.ngf,
                                     opt.norm,
                                     opt.init_type,
                                     opt.init_gain,
                                     self.gpu_ids)

        self.softmax = torch.nn.Softmax(dim=1)

        if self.isTrain:
            # define loss functions

            #opt.loss_mode=focal
            if self.opt.loss_mode == 'focal':
                self.criterionSeg = BinaryFocalLoss()   #设置损失函数为修正后的交叉熵损失函数，没有理解
            else:
                self.criterionSeg = nn.BCEWithLogitsLoss(size_average=True, reduce=True,
                    pos_weight=torch.tensor(1.0/3e-2,dtype=torch.float).to(self.device))
            self.weight_side = [0.5, 0.75, 1.0, 0.75, 0.5]  #这是论文里的αm 就是每个侧边输出层的损失权重

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.初始化优化;调度器将由函数< basemodel .setup>自动创建。
            self.optimizer = torch.optim.SGD(self.netG.parameters(), lr=opt.lr, momentum=0.9, weight_decay=2e-4)   # opt.lr=0.0001 使用随机梯度下降来进行参数更新
            self.optimizers.append(self.optimizer)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps. 从dataloader解压输入数据并执行必要的预处理步骤。
        Parameters:
            input (dict): include the data itself and its metadata information. 包含数据本身及其元数据信息。
        """
        # 数据加载详见deepcrack_dataset.py 中的getItem()
        self.image = input['image'].to(self.device)
        self.label = input['label'].to(self.device)

        self.image_paths = input['A_paths']

    # called by both functions < optimize_parameters > and < test >.
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # side_output1, side_output2, side_output3, side_output4, side_output5, fused
        self.outputs = self.netG(self.image) #把图片传入deepcrackNetwork结构，进行前向传播

        # for visualization
        self.label_viz = (self.label.float()-0.5)/0.5  #将label转换到（-1，1）区间
        self.fused = (torch.sigmoid(self.outputs[-1])-0.5)/0.5    #将fused层转换到（-1.1）区间
        # 将5个侧边输出特征图转换到（-1.1）区间存起来
        if self.display_sides:
            self.side1 = (torch.sigmoid(self.outputs[0])-0.5)/0.5
            self.side2 = (torch.sigmoid(self.outputs[1])-0.5)/0.5
            self.side3 = (torch.sigmoid(self.outputs[2])-0.5)/0.5
            self.side4 = (torch.sigmoid(self.outputs[3])-0.5)/0.5
            self.side5 = (torch.sigmoid(self.outputs[4])-0.5)/0.5

    def backward(self):
        """Calculate the loss"""
        lambda_side = self.opt.lambda_side   #self.opt.lambda_side=1.0 是侧输出损失的权重
        lambda_fused = self.opt.lambda_fused #self.opt.lambda_fused=1.0 是fused层输出损失的权重

        self.loss_side = 0.0
        # self.weight_side = [0.5, 0.75, 1.0, 0.75, 0.5]  #这是论文里的αm 就是每个侧边输出层的损失权重
        for out, w in zip(self.outputs[:-1], self.weight_side):
            self.loss_side += self.criterionSeg(out, self.label.float()) * w    #计算每个侧输出层的损失的总和

        self.loss_fused = self.criterionSeg(self.outputs[-1], self.label.float())  #计算fused层的损失
        self.loss_total = self.loss_side * lambda_side + self.loss_fused * lambda_fused    #计算总损失
        self.loss_total.backward()   #反向传播计算梯度

    def optimize_parameters(self, epoch=None):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute predictions. 前向传播
        self.optimizer.zero_grad()  # set G's gradients to zero 把梯度置为0
        self.backward()             # calculate gradients for G 反向传播，求梯度
        self.optimizer.step()       # update G's weights  使用随机梯度下降算法对梯度进行调优
