# -*- coding: utf-8 -*-
'''
此项目为FlyAI2.0新版本框架，数据读取，评估方式与之前不同
2.0框架不再限制数据如何读取
样例代码仅供参考学习，可以自己修改实现逻辑。
模版项目下载支持 PyTorch、Tensorflow、Keras、MXNET、scikit-learn等机器学习框架
第一次使用请看项目中的：FlyAI2.0竞赛框架使用说明.html
使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
学习资料可查看文档中心：https://doc.flyai.com/
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
'''


# 两块差别 ： 1.图像处理上面，没做归一化 2. 网络自己随机初始化权重，没有用与训练模型，随机初始化很可能在数据上不收敛的。如果出事不好则无法收敛，用预训练模型提供比较好的权重，不会乱走
# 100M参数，里面几百万维参数分布，随机初始化比较难学
# 为了解决问题所在、
# 这两块在书本上都碰不到的

import argparse
import os

from flyai.utils import remote_helper
from flyai.data_helper import DataHelper
from flyai.framework import FlyAI

from path import MODEL_PATH

import csv
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data as data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision

import copy


if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# 项目的超参，不使用可以删除
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")  #
parser.add_argument("-b", "--BATCH", default=4, type=int, help="batch size")  # batchsize和epochs都需要进行改 4的指数
args = parser.parse_args()

# 定义超参
IMG_SIZE = 256  # 图片尺寸不固定，统一resize 256*256
INPUT_SIZE = 224  # 随机裁剪224*224
BATCH_SIZE = args.BATCH  
EPOCHS = args.EPOCHS
BASE_LR = 0.01  # 学习率不是一成不变，有几种衰减方式，EPOCH STEP  STEP ：每训练一次一个step epoch：所有训练集过一遍  学习率会随着两个参数变化，可以定义函数变化
CUDA = torch.cuda.is_available()  # 界定本地跑还是GPU
DEVICE = torch.device("cuda" if CUDA else "cpu")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 归一化很重要，变成0~1，更有利于网络收敛
                                 std=[0.229, 0.224, 0.225])

train_transforms = transforms.Compose([  # 数据增强，numpy矩阵，最后转换为tensor向量 旋转可以去掉，第一次跑baseline可能70~80准确率
    transforms.Resize(IMG_SIZE),
    transforms.RandomResizedCrop(INPUT_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    normalize
])


class CovDataset(data.Dataset):
    def __init__(self, root, transform=None):
        img_root = root + "/image"
        train_root = root + "/train.csv"
        self.imgs_path = [os.path.join(img_root, k) for k in os.listdir(img_root)]
        self.imgs_key = os.listdir(img_root)
        self.transform = transform

        self.labels = self.__getlabel__(train_root)

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        img = Image.open(img_path).convert('RGB')  # 图像直接转换为RGB，绝对不能转换为灰度图，因为默认编码是RGB\RGB ALPHA，转为灰度信息丢失很多，变为3通道简单复制三份，相当于RGB -> 灰度图有一个计算公式。   灰度图很少用，进行分割的时候，分割标签是一个灰度图，自然图像
        if self.transform:
            img = self.transform(img)
        return img, self.labels[self.imgs_key[index]]

    def __getlabel__(self, trainfile):
        results ={}
        with open(trainfile, 'r') as trainCsv:
            lines = csv.reader(trainCsv)
            for line in lines:
                if line[1] == 'label':
                    continue
                key = line[0].split('/')[1]
                results[key] = int(line[1])
        return results

    def __len__(self):
        return len(self.imgs_key)


def Resnet50():  # 另一个比较关键的地方 一定要用别人论文已经用好的网络 工业应用ResNet极其以后的,最低ResNet。VGG Inception不用尝试了
    path = remote_helper.get_remote_date("https://www.flyai.com/m/resnet50-19c8e357.pth")  # 直接加载imagenet预训练模型
    model = torchvision.models.resnet50(pretrained=False)
    model.load_state_dict(torch.load(path))

    # 先将所有的特征层freeze
    for param in model.parameters():   # 将模型所有参数进行固定
        param.requires_grad = False

    # 放开想学习的层 可以通过查看model变量 # 放开要学习的参数  整个ResNet分四个block
    for param in model.layer4.parameters():  # 将最后一个block也让学习，这里是需要调整的。放到GPU要取消注释。不光要学最后一个block，和最后两个block
        param.requires_grad = True
    num_fc_ftr = model.fc.in_features   # 特征提取层、全连接层  # 之前为特征提取，最后分类任务，全连接进行分类  # 对应20
    model.fc = nn.Linear(num_fc_ftr, 2)  #  全连接两分类，之前的代码不是在CPU跑得，没有GPU
    model = model.to(DEVICE)  # 转到CUDA
    return model

# 不建议替代，可以看下 torch.optim.lr_scheduler.ReduceLROnPlateau(     这个怎么用，imagenet原版resnet是这么写的
def adjust_learning_rate(optimizer, epoch, lr):  # 调整学习率，每过15个epoch衰减为0.01，这些都需要对应调整 先来看训练效果，真正工程中两种方式，一种以0.01一直来学，波动比较大的话代表降学习率，手动方式。 第二种知道大体多少个epoch来调整，有很多方式。warmup，closing，比较多的点。学习率是个非常总要的调参数
    lr = lr * (0.1 ** (epoch // 15))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, device, train_loader, epoch, optimizer, criterion):  # 训练代码
    model.train()
    for batch_idx, data in enumerate(train_loader):
        x, y = data
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        torch.set_grad_enabled(True)
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        if batch_idx%50==0:
            print('Train Epoch: {}\t {}\t Loss: {:.6f}'.format(epoch, batch_idx, loss.item()))
    return model

# 为了不破坏前一个train方法结构，另起一个。其实可以在里面用if和传入phrase变量来控制
def eval(model, pre_model, device, eval_loader, len_dataset, optimizer, best_acc):  # 验证代码
    running_corrects = 0
    model.eval()
    for batch_idx, data in enumerate(eval_loader):
        x, y = data
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        torch.set_grad_enabled(False)
        y_hat = model(x)
        _, pred = torch.max(y_hat, 1)
        # print(pred)
        # print(y)
        running_corrects += torch.sum(pred == y)
    epoch_acc = running_corrects.double() / len_dataset
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        return model, best_acc
    return pre_model, best_acc


# epoch训练集准确率 和epoch次数一个曲线关系
# log打印出来  分类都会拟合的，除非训练集噪声特别多。  训练集不一定是100%，训练集一般在95左右~100%之间

class Main(FlyAI):
    '''
    项目中必须继承FlyAI类，否则线上运行会报错。
    '''

    def download_data(self):
        # 根据数据ID下载训练数据
        data_helper = DataHelper()
        data_helper.download_from_ids("COVIDClassification")

    def deal_with_data(self):
        '''
        处理数据，没有可不写。
        :return:
        '''
        pass

    def train(self):
        '''
        训练模型，必须实现此方法
        :return:
        '''
        dataset = CovDataset('./data/input/COVIDClassification', transform=train_transforms)
        train_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        print(len(dataset))  # 训练集有多少不知道，要看训练日志。训练集多少如果只有1k、2k张，epoch多一些。信息很重要。

        model = Resnet50()
        if not CUDA:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        optimizer = torch.optim.SGD(model.parameters(), BASE_LR, momentum=0.9, weight_decay=0.0001)  # 这两个都是默认的，不用调0.9和10e-4 参考数值 weight_decay L2正则，作用就是LR正则项，抑制学习范围。100维空间负无穷到正无穷。-10 ~ +10

        eval_loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=BATCH_SIZE, shuffle=True
        )

        best_acc = 0.0
        for epoch in range(EPOCHS):
            adjust_learning_rate(optimizer, epoch, BASE_LR)
            curr_model = train(model, DEVICE, train_loader, epoch, optimizer, criterion)
            # 只返回更优的model
            model, best_acc = eval(curr_model, model, DEVICE, eval_loader, len(dataset), optimizer, best_acc)


if __name__ == '__main__':
    main = Main()
    # main.download_data()
    main.train()
