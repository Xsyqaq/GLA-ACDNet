import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
class DAM_Channel(nn.Module):
    """Channel Attention Module (CAM)"""

    def __init__(self):
        super().__init__()
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        proj_query = x.view(batch_size, channels, -1)
        proj_key = x.view(batch_size, channels, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = F.softmax(energy_new, dim=-1)
        proj_value = x.view(batch_size, channels, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, channels, height, width)

        out = self.gamma * out + x  # 使用 * 操作符而不是 () 调用
        return out


class DAM_Position(nn.Module):
    def __init__(self, in_channels):
        super(DAM_Position, self).__init__()
        self.gama = nn.Parameter(torch.ones(1))  # 初始化1

        # 定义深度可分离卷积层
        self.sep_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 1, kernel_size=1, stride=1, padding=0, groups=in_channels),  #默认除以是8
            nn.ReLU(inplace=True)
        )
        self.sep_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 获取输入特征图的shape
        b, c, h, w = x.size()

        # 深度可分离卷积
        x1 = self.sep_conv1(x)
        # 调整维度排序
        x1_trans = x1.permute(0, 2, 3, 1).reshape(b, c // 1, h * w) #这里也要改除数
        x1_trans_reshape_trans = x1_trans.permute(0, 2, 1)
        # 矩阵相乘
        x1_mutmul = torch.bmm(x1_trans_reshape_trans, x1_trans)
        # 经过softmax归一化权重
        x1_mutmul = F.softmax(x1_mutmul, dim=-1)

        # 深度可分离卷积
        x2 = self.sep_conv2(x)
        x2_trans_reshape = x2.permute(0, 2, 3, 1).reshape(b, c, h * w)

        # 调整x1_mutmul的轴，和x2矩阵相乘
        x1_mutmul_trans = x1_mutmul.permute(0, 2, 1)
        x2_mutmul = torch.bmm(x2_trans_reshape, x1_mutmul_trans)
        x2_mutmul = x2_mutmul.reshape(b, c, h, w)

        # 结果乘以可训练变量
        x2_mutmul = x2_mutmul * self.gama
        # 输入和输出叠加
        x = x2_mutmul + x

        return x

