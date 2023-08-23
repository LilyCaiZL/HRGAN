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
os.makedirs("images16", exist_ok=True)
os.makedirs("saved_models16", exist_ok=True)

# 然后设置初始参数，并打印
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
# parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
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
class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_residual_blocks=16):
        super(GeneratorResNet, self).__init__()

        # 第一个卷积层，见注释１
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), nn.PReLU())

        # 初始化残差块
        res_blocks = []

        # 生成16个残差块，见注释２
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # 第二个卷积层，见注释３
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8))

        # 上采样层，见注释４
        upsampling = []
        for out_features in range(1):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # 第三个卷积层，见注释４
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4), nn.Tanh())

    def forward(self, x):
        # print(x)
        out1 = self.conv1(x)

        out2 = self.res_blocks(out1)
        out3 = self.conv2(out2)
        out4 = torch.add(out1, out3)
        out5 = self.upsampling(out4)
        out = self.conv3(out5)

        return out


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
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
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

generator = GeneratorResNet()
discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
feature_extractor = FeatureExtractor()

# 将特征提取器设为评估模式
feature_extractor.eval()

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
    generator.load_state_dict(torch.load("saved_models16/generator_%d.pth"))
    discriminator.load_state_dict(torch.load("saved_models16/discriminator_%d.pth"))

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
        self.filesH = sorted(os.listdir('/data/DeepRockSR-2D_copy/DeepRockSR-2D/shuffled2D/shuffled2D_train_HR'))
        self.filesL = sorted(os.listdir('/data/DeepRockSR-2D_copy/DeepRockSR-2D/shuffled2D/shuffled2D_train_HRB_G_BB9'))

    def __getitem__(self, index):  # 定义时未调用，每次读取图像时调用，见注释9
        imgl = Image.open('/data/DeepRockSR-2D_copy/DeepRockSR-2D/shuffled2D/shuffled2D_train_HRB_G_BB9/' + self.filesL[
            index % len(self.filesL)]).convert('L')
        imgh = Image.open('/data/DeepRockSR-2D_copy/DeepRockSR-2D/shuffled2D/shuffled2D_train_HR/' + self.filesH[
            index % len(self.filesH)]).convert('L')
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
            # save_image(img_grid, "images16/%d.png" % batches_done, normalize=False)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # 每10个epoch保存一次模型
        torch.save(generator.state_dict(), "saved_models16/generator_%d.pth" % epoch)
        torch.save(discriminator.state_dict(), "saved_models16/discriminator_%d.pth" % epoch)

