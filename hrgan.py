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
torch.cuda.set_device(0)
IMAGE_NAME='images_hrnet'
SAVE_MODEL_DIR='saved_models_hrnet'
os.makedirs(IMAGE_NAME, exist_ok=True)
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

# 然后设置初始参数，并打印
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
# parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=500, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=500, help="high res. image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available()

hr_shape = (opt.hr_height, opt.hr_width)
import torch.nn as nn
import torch.nn.functional as F
import torch
# from torchvision.models import vgg19
from torchvision.models import densenet121
import math
import os
import logging


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
        self.convtran2d = nn.ConvTranspose2d(1,32,kernel_size=5,stride=2,padding=1,bias=False,output_padding=1)
        self.bn1 = self.norm_layer(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1,
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


# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, input_shape): # input_shape: (3, 128, 128)
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape


        # 这里有坑需要解决
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h+1, patch_w+1) # patch_h: 8 patch_w: 8

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = [] # 每次layers都会重置为空
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:    # 当first_block为False时，layers的第二层添加一个BN层
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = [] # 见注释６
        in_filters = in_channels # in_filters = 3
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        # 见注释７
        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

from collections import OrderedDict
# 定义特征提取器
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = densenet121(pretrained=False)
        # vgg19_model.conv0=nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        vgg19_model.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, 64, kernel_size=7, stride=2,
                                padding=2, bias=False)),
            ('norm0', nn.BatchNorm2d(64)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        # 取了VGG19的前18层，见注释8
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:120])

    def forward(self, img):
        return self.feature_extractor(img)

generator = GeneratorResNet(cfg)
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
feature_extractor = FeatureExtractor()

# 将特征提取器设为评估模式
feature_extractor.eval()
generator.eval()
# 设置损失函数，MSELoss和L1Loss
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

# if cuda:
generator = generator.cuda()
discriminator = discriminator.cuda()
feature_extractor = feature_extractor.cuda()
criterion_GAN = criterion_GAN.cuda()
criterion_content = criterion_content.cuda()
# 从第2次循环开始，载入训练得到的生成器和判别器模型
if opt.epoch != 0:
    generator.load_state_dict(torch.load(SAVE_MODEL_DIR+"/generator_%d.pth"))
    discriminator.load_state_dict(torch.load(SAVE_MODEL_DIR+"/discriminator_%d.pth"))

# 设置优化器
# 生成器的优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# 判别器的优化器
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

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
        # self.files = sorted(glob.glob(root + "/*.*"))
        self.filesH = sorted(os.listdir('/data/lilycai/DeepRockSR-2D_copy/DeepRockSR-2D/shuffled2D/shuffled2D_train_HR'))
        self.filesL = sorted(os.listdir('/data/lilycai/DeepRockSR-2D_copy/DeepRockSR-2D/shuffled2D/shuffled2D_train_HRB_G_BB9'))

    def __getitem__(self, index): # 定义时未调用，每次读取图像时调用，见注释9
        imgl = Image.open('/data/lilycai/DeepRockSR-2D_copy/DeepRockSR-2D/shuffled2D/shuffled2D_train_HRB_G_BB9/'+self.filesL[index % len(self.filesL)]).convert('L')
        imgh = Image.open('/data/lilycai/DeepRockSR-2D_copy/DeepRockSR-2D/shuffled2D/shuffled2D_train_HR/'+self.filesH[index % len(self.filesH)]).convert('L')
        img_lr = self.lr_transform(imgl)
        img_hr = self.hr_transform(imgh)

        # img_lr = imgl
        # img_hr = imgh

        return {"lr": img_lr, "hr": img_hr}

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

#定义Tensor类型
# Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
Tensor = torch.cuda.FloatTensor

print('start training')
try:
    os.mkdir('images_hrnet2')
    os.mkdir('saved_models_hrnet2')
except:
    pass
for epoch in range(opt.epoch, opt.n_epochs):
    print(epoch)
    for i, imgs in enumerate(dataloader):
        print(i)
        # 定义低、高分辨率图像对，imgs为字典
        imgs_lr = Variable(imgs["lr"].type(Tensor)) # torch.Size([4,3,500,500])
        imgs_hr = Variable(imgs["hr"].type(Tensor)) # torch.Size([4,3,500,500])

        # 生成地面真值,真为1，假为0
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

        # torch.Size([4,1,8,8])
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
        # torch.Size([4,1,8,8])

        # ------------------
        #  训练生成器
        # ------------------
        print('训练生成器')
        optimizer_G.zero_grad()
        print('利用生成器从低分辨率图像生成高分辨率图像')
        # 利用生成器从低分辨率图像生成高分辨率图像，见注释10
        gen_hr = generator(imgs_lr) # gen_hr: (4,3,500,500)
        print('gen_hr.shape',gen_hr.shape)
        # 对抗损失，见注释11
        # 第一次循环: tensor(0.9380, device='cuda:0', grad_fn=<MseLossBackward>)
        print('对抗损失,第一次循环')
        k=discriminator(gen_hr)
        print('维度',k.shape,valid.shape)

        loss_GAN = criterion_GAN(k, valid)

        # 内容损失，见注释12
        print('内容损失')
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach())

        # 生成器的总损失
        print('生成器的总损失')
        loss_G = loss_content + 1e-3 * loss_GAN

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  训练判别器
        # ---------------------
        print('训练判别器')
        optimizer_D.zero_grad()

        # 真假图像的损失
        print('真假图像的损失')
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

        # 判别器的总损失
        print('判别器的总损失')
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  输出记录
        # --------------

        sys.stdout.write(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())+'\n'
        ) # 相当于print()
        with open('tmp.txt','a') as f:
            f.write("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())+'\n'
        )
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            #保存上采样和SRGAN输出的图像
            imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
            gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
            imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
            # img_grid = torch.cat((imgs_lr, gen_hr), -1)
            # save_image(img_grid, IMAGE_NAME+"/%d.png" % batches_done, normalize=False)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # 每10个epoch保存一次模型
        torch.save(generator.state_dict(), SAVE_MODEL_DIR+"/generator_%d.pth" % epoch)
        torch.save(discriminator.state_dict(), SAVE_MODEL_DIR+"/discriminator_%d.pth" % epoch)

