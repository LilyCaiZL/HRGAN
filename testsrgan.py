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

# 定义特征提取器
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = densenet121(pretrained=False)
        # 取了VGG19的前18层，见注释８
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:120])

    def forward(self, img):
        return self.feature_extractor(img)

generator = GeneratorResNet()
# discriminator = Discriminator(input_shape=(opt.channels, *hr_shape))
feature_extractor = FeatureExtractor()

# 将特征提取器设为评估模式
feature_extractor.eval()

# 设置损失函数，MSELoss和L1Loss
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

generator.load_state_dict(torch.load("saved_models16/generator_"+str(opt.gennum)+".pth"))
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
        # self.files = sorted(glob.glob(root + "/*.*"))
        self.filesH = sorted(os.listdir('/data/DeepRockSR-2D_copy/DeepRockSR-2D/shuffled2D/shuffled2D_valid_HR'))
        self.filesL = sorted(os.listdir('/data/DeepRockSR-2D_copy/DeepRockSR-2D/shuffled2D/shuffled2D_valid_HRB_G_BB9'))

    def __getitem__(self, index): # 定义时未调用，每次读取图像时调用，见注释9
        imgl = Image.open('/data/DeepRockSR-2D_copy/DeepRockSR-2D/shuffled2D/shuffled2D_valid_HRB_G_BB9/'+self.filesL[index % len(self.filesL)]).convert('L')
        imgh = Image.open('/data/DeepRockSR-2D_copy/DeepRockSR-2D/shuffled2D/shuffled2D_valid_HR/'+self.filesH[index % len(self.filesH)]).convert('L')
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
    os.mkdir(str(opt.gennum)+'imageval')
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
        gnn = gn[k]*255
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

        im.save(str(opt.gennum)+'imageval/'+str(i)+str(k)+".png")

        # 四倍下采样和模糊后的图像
        irr = ir[k]*255
        irr = irr[0]
        # irr = irr.transpose(1,0)
        # print('irr',irr.shape)
        im2 = Image.fromarray(np.uint8(irr))
        im2.save(str(opt.gennum)+'imageval/' + str(i) + str(k) + "-.png")

        # 原图
        hrr = hr[k] * 255
        hrr=hrr[0]
        # hrr = hrr.transpose(1,0)
        im3 = Image.fromarray(np.uint8(hrr))
        im3.save(str(opt.gennum)+'imageval/' + str(i) + str(k) + "--.png")

        # img1 = cv2.imread(str(opt.gennum)+'imageval/'+str(i)+str(k)+".png")
        #
        # img2 = cv2.imread(str(opt.gennum)+'imageval/' + str(i) + str(k) + "--.png")
        #
        # print('PSNR shape',img1.shape,img2.shape)
        # PSNR = peak_signal_noise_ratio(img1, img2)
        # SSIM = compare_ssim(img1, img2, multichannel=True)
        #
        # PSNRlist.append(PSNR)
        # SSIMlist.append(SSIM)
        #
        # print(PSNR,SSIM)
# print(str(sum(mselist)/len(mselist)))
# with open('result/'+str(opt.gennum)+'result-srgan.txt','w') as f:
#     f.write('\nloss:'+str(sum(mselist)/len(mselist)))
#     f.write('\nPSNR:'+str(sum(PSNRlist)/len(PSNRlist)))
#     f.write('\nSSIM:'+str(sum(SSIMlist)/len(SSIMlist)))
#     print(sum(mselist)/len(mselist))
