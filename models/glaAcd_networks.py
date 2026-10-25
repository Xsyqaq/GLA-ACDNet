import torch
import torch.nn.functional as F
from models.networks import get_norm_layer, init_net
from models.attention.ELA import ELA
from torch import nn
from models.attention.SE import SE_Block

class MSSC(nn.Module):
    def __init__(self, dim,outdim,branch_ratio=0.25):
        super(MSSC,self).__init__()
        gc = int(outdim * branch_ratio)
        self.conv0 = nn.Conv2d(dim, outdim, 3, padding=1,stride=1)
        self.conv0_1 = nn.Conv2d(gc, gc, (1, 3), padding=(0, 1), groups=gc)
        self.conv0_2 = nn.Conv2d(gc, gc, (3, 1), padding=(1, 0), groups=gc)

        self.conv1_1 = nn.Conv2d(gc,gc, (1, 5), padding=(0, 2), groups=gc)
        self.conv1_2 = nn.Conv2d(gc,gc, (5, 1), padding=(2, 0), groups=gc)

        self.conv2_1 = nn.Conv2d(gc, gc, (1, 7), padding=(0, 3), groups=gc)
        self.conv2_2 = nn.Conv2d(gc, gc, (7, 1), padding=(3, 0), groups=gc)
        self.bn = nn.BatchNorm2d(outdim)
        self.ReLU = nn.ReLU(True)
        self.split_indexes = (outdim-3*gc,gc,gc,gc)
        self.SE = SE_Block(gc,8)
        self.conv3 = nn.Conv2d(outdim, outdim, 1)

    def forward(self, x):
        #u = x.clone()
        attn = self.conv0(x)
        x_res,attn_0,attn_1,attn_2= torch.split(attn, self.split_indexes, dim=1)
        attn_0 = self.conv0_1(attn_0)
        attn_0 = self.conv0_2(attn_0)
        attn_1 = self.conv1_1(attn_1)
        attn_1 = self.conv1_2(attn_1)
        attn_2 = self.conv2_1(attn_2)
        attn_2 = self.conv2_2(attn_2)

        x_res = self.SE(x_res)
        attn_0 = self.SE(attn_0)
        attn_1 = self.SE(attn_1)
        attn_2 = self.SE(attn_2)
        attn_new = torch.cat([x_res,attn_0,attn_1,attn_2],dim=1)
        attn_new = self.conv3(attn_new)
        return self.ReLU(self.bn(attn+attn_new))

class CLKC(nn.Module):
    def __init__(self, inch, outch, concatoutch):
        super(CLKC, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(inch, outch, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.BatchNorm2d(outch),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(outch, outch, kernel_size=3, stride=1, dilation=3, padding=3),
            nn.BatchNorm2d(outch),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(outch, outch, kernel_size=3, stride=1, dilation=6, padding=6),
            nn.BatchNorm2d(outch),
            nn.ReLU(),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(outch, outch, kernel_size=3, stride=1, dilation=9, padding=9),
            nn.BatchNorm2d(outch),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(4*outch+inch, concatoutch, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(concatoutch),
            nn.ReLU()
        )
    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        out = self.layer5(torch.cat([x, layer1, layer2, layer3, layer4], dim=1))
        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Block(nn.Sequential):
    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True, norm_layer=nn.BatchNorm2d):
        super(Block, self).__init__()
        if bn_start:
            self.add_module('norm1', norm_layer(input_num)),

        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)),

        self.add_module('norm2', norm_layer(num1)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                                           dilation=dilation_rate, padding=dilation_rate)),
        self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(Block, self).forward(_input)
        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)
        return feature


def Upsample(x, size, align_corners=False):
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=align_corners)


class ContextBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio,
                 pooling_type='avg',
                 fusion_types=('channel_mul',)):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pooling_type == 'att':
            # kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out + out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out

class ConvBranch(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        res1 = x
        res2 = x
        x = self.conv1(x)
        x = x + self.conv2(x)
        x = self.conv3(x)
        x = x + self.conv4(x)
        x = self.conv5(x)
        x = x + self.conv6(x)
        x = self.conv7(x)
        x_mask = self.sigmoid_spatial(x)
        res1 = res1 * x_mask
        return res2 + res1

class GLSA(nn.Module):

    def __init__(self, input_dim=5, embed_dim=5, k_s=3):
        super().__init__()

        self.conv1_1 = BasicConv2d(embed_dim * 2, embed_dim, 1)
        self.local_11conv = nn.Conv2d(input_dim, embed_dim, 1)
        self.global_11conv = nn.Conv2d(input_dim, embed_dim, 1)
        self.GlobelBlock = ContextBlock(inplanes=embed_dim, ratio=2)
        self.local = ConvBranch(in_features=embed_dim, hidden_features=embed_dim, out_features=embed_dim)

    def forward(self, x):
        b, c, h, w = x.size()

        # local block
        local = self.local(self.local_11conv(x))

        # Global block
        Globel = self.GlobelBlock(self.global_11conv(x))

        # concat Global + local
        x = torch.cat([local, Globel], dim=1)
        x = self.conv1_1(x)

        return x

class DeepCrackNet(nn.Module):
    def __init__(self, in_nc=3, num_classes=1, ngf=32, norm='batch'):
        super(DeepCrackNet, self).__init__()
        "ngf参数是outputs_channels"
        norm_layer = get_norm_layer(norm_type=norm)
        self.conv1 = nn.Sequential(*self._conv_block(in_nc, ngf, norm_layer, num_block=2))
        self.side_conv1 = nn.Conv2d(ngf, num_classes, kernel_size=1, stride=1, bias=False)

        self.conv2 = nn.Sequential(*self._conv_block(ngf, ngf*2, norm_layer, num_block=2))
        self.side_conv2 = nn.Conv2d(ngf*2, num_classes, kernel_size=1, stride=1, bias=False)

        self.conv3 = nn.Sequential(*self._conv_block(ngf*2, ngf*4, norm_layer, num_block=3))
        self.side_conv3 = nn.Conv2d(ngf*4, num_classes, kernel_size=1, stride=1, bias=False)

        self.conv4 = nn.Sequential(*self._conv_block(ngf*4, ngf*8, norm_layer, num_block=3))
        self.side_conv4 = nn.Conv2d(ngf*8, num_classes, kernel_size=1, stride=1, bias=False)

        self.conv5 = nn.Sequential(*self._conv_block(ngf*8, ngf*8, norm_layer, num_block=3))
        self.side_conv5 = nn.Conv2d(ngf*8, num_classes, kernel_size=1, stride=1, bias=False)

        self.fuse_conv = nn.Conv2d(num_classes*5, num_classes, kernel_size=1, stride=1, bias=False)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.temp1 = ELA(in_channels=32, phi='T')
        self.temp2 = ELA(in_channels=64, phi='T')
        self.temp3 = ELA(in_channels=128, phi='S')
        self.temp4 = CLKC(256,64,256)
        self.temp5 = CLKC(256,64,256)
        self.end = GLSA(5,5)

    def _conv_block(self, in_nc, out_nc, norm_layer, num_block=2, kernel_size=3,
        stride=1, padding=1, bias=False):
        conv = []
        for i in range(num_block):
            cur_in_nc = in_nc if i == 0 else out_nc
            conv += [MSSC(cur_in_nc, out_nc)]
        return conv

    def forward(self, x):
        #h w 第三维和第四维 高，宽
        h,w = x.size()[2:]
        # main stream features
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.maxpool(conv1))
        conv3 = self.conv3(self.maxpool(conv2))
        conv4 = self.conv4(self.maxpool(conv3))
        conv5 = self.conv5(self.maxpool(conv4))
        # side output features
        side_output1 = self.side_conv1(self.temp1(conv1))
        side_output2 = self.side_conv2(self.temp2(conv2))
        side_output3 = self.side_conv3(self.temp3(conv3))
        side_output4 = self.side_conv4(self.temp4(conv4))
        side_output5 = self.side_conv5(self.temp5(conv5))

        # upsampling side output features
        side_output2 = F.interpolate(side_output2, size=(h, w), mode='bilinear', align_corners=True) #self.up2(side_output2)
        side_output3 = F.interpolate(side_output3, size=(h, w), mode='bilinear', align_corners=True) #self.up4(side_output3)
        side_output4 = F.interpolate(side_output4, size=(h, w), mode='bilinear', align_corners=True) #self.up8(side_output4)
        side_output5 = F.interpolate(side_output5, size=(h, w), mode='bilinear', align_corners=True) #self.up16(side_output5)
        fused = torch.cat([side_output1,
                                            side_output2,
                                            side_output3,
                                            side_output4,
                                            side_output5], dim=1)
        fused = self.end(fused) + fused
        fused = self.fuse_conv(fused)

        return side_output1, side_output2, side_output3, side_output4, side_output5, fused

def define_deepcrack(in_nc,
                     num_classes,
                     ngf,
                     norm='batch',
                     init_type='xavier',
                     init_gain=0.02,
                     gpu_ids=[]):
    net = DeepCrackNet(in_nc, num_classes, 32, norm)
    return init_net(net, init_type, init_gain, gpu_ids)


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.size_average = size_average
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.criterion(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.size_average:
            return F_loss.mean()
        else:
            return F_loss.sum()

if __name__=="__main__":
    input=torch.rand(1,3,256,256)
    model=DeepCrackNet(3,1,64,'batch')
    output=model(input)
    i=0
    while i<6:
        print(output[i].shape)
        i=i+1
