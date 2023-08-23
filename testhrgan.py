# 首先导入相关模块，并设置系统环境：

import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

# from models import *
# from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 设置系统可见的GPU从１号开始
torch.cuda.set_device(1)
os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

# 然后设置初始参数，并打印
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
# parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu",type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=500, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=500, help="high res. image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between model checkpoints")
parser.add_argument('--gennum',type=str)

opt = parser.parse_args()

print(opt)
print('gennum',opt.gennum)

# cuda = torch.cuda.is_available()


hr_shape = (opt.hr_height, opt.hr_width)
import torch.nn as nn
import torch.nn.functional as F
import torch
# from torchvision.models import vgg19
from torchvision.models import densenet121
import math

# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()  # in_features: 64
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)

# 定义生成器
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True, norm_layer=None):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(num_channels[branch_index] * block.expansion),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample, norm_layer=self.norm_layer))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index], norm_layer=self.norm_layer))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        self.norm_layer(num_inchannels[i])))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                self.norm_layer(num_outchannels_conv3x3),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear',
                        align_corners=True
                        )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}

cfg={'STAGE1':{'NUM_MODULES':1,'NUM_BRANCHES':1,'BLOCK':'BOTTLENECK','NUM_BLOCKS':[4],'NUM_CHANNELS':[32],'FUSE_METHOD':'SUM'},
     'STAGE2':{'NUM_MODULES':1,'NUM_BRANCHES':2,'BLOCK':'BASIC','NUM_BLOCKS':[4,4],'NUM_CHANNELS':[8,16],'FUSE_METHOD':'SUM'},
     'STAGE3':{'NUM_MODULES':3,'NUM_BRANCHES':3,'BLOCK':'BASIC','NUM_BLOCKS':[4,4,4],'NUM_CHANNELS':[8,16,32],'FUSE_METHOD':'SUM'},
     'STAGE4':{'NUM_MODULES':4,'NUM_BRANCHES':4,'BLOCK':'BASIC','NUM_BLOCKS':[4,4,4,4],'NUM_CHANNELS':[8,16,32,64],'FUSE_METHOD':'SUM'}}

# 定义生成器
class GeneratorResNet(nn.Module):
    def __init__(self,
                 cfg,
                 norm_layer=None):
        super(GeneratorResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer
        # stem network
        # stem net
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0,
        #                        bias=False)
        self.convtran2d = nn.ConvTranspose2d(1,32,kernel_size=5,stride=2,padding=2,bias=False,output_padding=1)
        self.bn1 = self.norm_layer(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2,
                               bias=False)
        self.bn2 = self.norm_layer(32)
        self.relu = nn.ReLU(inplace=True)

        # stage 1
        self.stage1_cfg = cfg['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 32, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels


        # stage 2
        self.stage2_cfg = cfg['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        # stage 3
        self.stage3_cfg = cfg['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        # stage 4
        self.stage4_cfg = cfg['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        last_inp_channels = np.int(np.sum(pre_stage_channels))

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            self.norm_layer(last_inp_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0)
        )

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        self.norm_layer(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        self.norm_layer(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_layer=self.norm_layer))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=self.norm_layer))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output,
                                     norm_layer=self.norm_layer)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        # print('xc1',x.shape)
        # x = self.conv1(x)
        x = self.convtran2d(x)
        # print('xc2', x.shape)
        x = self.bn1(x)
        # print('xc2', x.shape)
        x = self.relu(x)
        # print('xc2',x.shape)
        #
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        x = self.layer1(x)
        # print('xl1',x.shape)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
            print('stage2',x.shape)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])


        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        # print('x[0].shape',x[0].shape)
        # print('x1.shape',x1.shape)
        # print('x2.shape',x2.shape)
        # print('x3.shape',x3.shape)
        x = torch.cat([x[0], x1, x2, x3], 1)
        x = self.last_layer(x)
        # print('x.shape',x.shape)

        return x

from collections import OrderedDict

# 定义特征提取器
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = densenet121(pretrained=False)
        # vgg19_model.conv0=nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        vgg19_model.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, 64, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(64)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        # 取了VGG19的前18层，见注释8
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:120])

    def forward(self, img):
        return self.feature_extractor(img)

generator = GeneratorResNet(cfg)
# discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
feature_extractor = FeatureExtractor()

# 将特征提取器设为评估模式
feature_extractor.eval()

# 设置损失函数，MSELoss和L1Loss
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

generator.load_state_dict(torch.load("saved_models_hrnet/generator_170.pth"))
    # discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth"))
generator.eval()

# 导入数据集
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# 设定预训练PyTorch模型的归一化参数
# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])


class ImageDataset(Dataset):
    def __init__(self, hr_shape):
        hr_height, hr_width = hr_shape # hr_shape=(128, 128)
        # 通过源图像分别生成低、高分辨率图像，4倍
        self.lr_transform = transforms.Compose( # 见注释8
            [
                transforms.Resize((hr_height // 2, hr_height // 2), Image.BICUBIC),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                # transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                # transforms.Normalize(mean, std),
            ]
        )
        # 将文件夹中的图片进行按文件名升序排列，从000001.jpg到202599.jpg
        self.filesH = sorted(os.listdir('/data/DeepRockSR-2D_copy/DeepRockSR-2D/shuffled2D/shuffled2D_valid_HR'))
        self.filesL = sorted(os.listdir('/data/DeepRockSR-2D_copy/DeepRockSR-2D/shuffled2D/shuffled2D_valid_HRB_G_BB9'))

    def __getitem__(self, index):  # 定义时未调用，每次读取图像时调用，见注释9
        imgl = Image.open('/data/DeepRockSR-2D_copy/DeepRockSR-2D/shuffled2D/shuffled2D_valid_HRB_G_BB9/' + self.filesL[
            index % len(self.filesL)]).convert('L')
        imgh = Image.open('/data/DeepRockSR-2D_copy/DeepRockSR-2D/shuffled2D/shuffled2D_valid_HR/' + self.filesH[
            index % len(self.filesH)]).convert('L')
        # im2 = imgl
        # im2.save(str(index) + "-.png")

        img_lr = self.lr_transform(imgl)
        img_hr = self.hr_transform(imgh)


        return {"lr": img_lr,"hr": img_hr}

    # 定义dataloader和每次读取图像时均调用
    def __len__(self):
        return len(self.filesH)

# 用定义好的方法来读取数据集
dataloader = DataLoader(
    ImageDataset(hr_shape=hr_shape),
    batch_size=opt.batch_size,  # batch_size = 4
    shuffle=False,
    num_workers=opt.n_cpu, # num_workers = 8
)
print('数据集准备完毕')

# 定义特征提取器


feature_extractor = FeatureExtractor()
feature_extractor.eval()

#定义Tensor类型
# Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
Tensor = torch.FloatTensor

from PIL import Image
# import numpy
# print('start training')

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2
from skimage.metrics import peak_signal_noise_ratio
from skimage.measure import compare_ssim


mselist = [0 for _ in range(len(dataloader))]
PSNRlist =[]
SSIMlist = []

try:
    os.mkdir(str(opt.gennum)+'imageval_hrnet/')
    os.mkdir('result')
except:
    pass
for i,imgs in enumerate(dataloader):
    # print(i)

    imgs_lr = Variable(imgs["lr"].type(Tensor)) # torch.Size([4,3,500,500])
    imgs_hr = Variable(imgs["hr"].type(Tensor))
    gen_hr = generator(imgs_lr)
    gn = gen_hr.detach().numpy()
    ir = imgs_lr.detach().numpy()
    hr = imgs_hr.detach().numpy()

    # print('gn.shape',gn.shape)

    gen_features = feature_extractor(gen_hr)
    real_features = feature_extractor(imgs_hr)
    print('gen_features.shape', gen_features.shape)
    print('real_features.shape', real_features.shape)
    loss_content = criterion_content(gen_features, real_features.detach())
    print('loss_content',loss_content)
    mselist[i]=loss_content.item()
    print('loss_content',mselist[i])
    print('gnlen',len(gn))
    print('gnshape',gn.shape)
    for k in range(len(gn)):
        gnn = gn[k]
        # gnn = gnn.astype(np.uint8)
        # gnn = gnn.reshape(500,500,3)

        # gnn=np.array()
        # gnn = gnn.transpose(1,0)
        # gnn = np.resize(gnn, (500, 500, 3))
        gnn = gnn[0]
        # print('gnnshape',gnn.shape)
        # gnn = gnn.reshape(gnn, (500, 500))
        # print('gnn',gnn.shape)

        # 生成的图像
        im = Image.fromarray(np.uint8(gnn))

        im.save(str(opt.gennum)+'imageval_hrnet/'+str(i)+str(k)+".png")

        # 四倍下采样和模糊后的图像
        irr = ir[k]*255
        irr = irr[0]
        # irr = irr.transpose(1,0)
        # print('irr',irr.shape)
        im2 = Image.fromarray(np.uint8(irr))
        im2.save(str(opt.gennum)+'imageval_hrnet/' + str(i) + str(k) + "-.png")

        # 原图
        hrr = hr[k] * 255
        hrr=hrr[0]
        # hrr = hrr.transpose(1,0)
        im3 = Image.fromarray(np.uint8(hrr))
        im3.save(str(opt.gennum)+'imageval_hrnet/' + str(i) + str(k) + "--.png")

        img1 = cv2.imread(str(opt.gennum)+'imageval_hrnet/'+str(i)+str(k)+".png")

        img2 = cv2.imread(str(opt.gennum)+'imageval_hrnet/' + str(i) + str(k) + "--.png")

        print('PSNR shape',img1.shape,img2.shape)
        PSNR = peak_signal_noise_ratio(img1, img2)
        SSIM = compare_ssim(img1, img2, multichannel=True)

        PSNRlist.append(PSNR)
        SSIMlist.append(SSIM)

        print(PSNR,SSIM)

print(str(sum(mselist)/len(mselist)))
with open('result/'+str(opt.gennum)+'result_hrnet.txt','a') as f:
    f.write('\nloss:'+str(sum(mselist)/len(mselist)))
    f.write('\nPSNR:'+str(sum(PSNRlist)/len(PSNRlist)))
    f.write('\nSSIM:'+str(sum(SSIMlist)/len(SSIMlist)))

    print(sum(mselist)/len(mselist))