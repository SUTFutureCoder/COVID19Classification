# -*- coding: utf-8 -*-
import argparse
import os

from PIL import Image
from flyai.data_helper import DataHelper
from flyai.framework import FlyAI

from path import MODEL_PATH

import csv

import torch
import torch.utils
import torch.utils.data.dataset
import torch.nn as nn
from torchvision import transforms
import numpy as np


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
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# 项目的超参，不使用可以删除
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=8, type=int, help="batch size")
args = parser.parse_args()


class SimpleNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.model1 = nn.Sequential(
            SimpleNet(3, 32),
            SimpleNet(32, 32),
            SimpleNet(32, 32)
        )

        self.extra1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32)
        )

        self.model2 = nn.Sequential(
            SimpleNet(32, 64),
            SimpleNet(64, 64),
            SimpleNet(64, 64),
            SimpleNet(64, 64),
        )

        self.extra2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64)
        )


        self.model3 = nn.Sequential(
            SimpleNet(64, 128),
            SimpleNet(128, 128),
            SimpleNet(128, 128),
            SimpleNet(128, 128),
        )

        self.extra3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128)
        )

        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(4)

        self.fc = nn.Linear(in_features=16*16*128, out_features=2)  # 输出0、1

    def forward(self, x):
        out = self.model1(x) + self.extra1(x)
        out = self.maxpool(out)  # 256 -> 128
        out = self.model2(out) + self.extra2(out)
        out = self.maxpool(out)  # 128 -> 64
        out = self.model3(out) + self.extra3(out)
        out = self.avgpool(out)  # 64 -> 16
        out = out.view(-1, 16*16*128)
        out = self.fc(out)
        return out


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(3),  # 4通道图像统一转为3通道
    transforms.ToTensor(),
])

class FlameSet(torch.utils.data.Dataset):
    def __init__(self, root):
        img_root = root + "/image"
        imgs = os.listdir(img_root)
        self.imgs = [os.path.join(img_root, k) for k in imgs]

        train_root = root + "/train.csv"
        self.file = [k for k in imgs]

        # 读取train.csv
        self.label_map = {}
        self.__getlabel__(train_root)

        self.transforms = transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)

        nn.init.kaiming_normal_(data)  # 非常大的问题，kaiming normal api 和data没有关系，没有对data进行归一化 。可以用在参数初始化上面，定义网络层有学习参数，对进行初始化，要比高斯其他方法更好一些
        return data, self.label_map["image/"+self.file[index]]

    def __getlabel__(self, trainfile):
        with open(trainfile, 'r') as trainCsv:
            lines = csv.reader(trainCsv)
            for line in lines:
                if line[1] == 'label':
                    continue
                self.label_map[line[0]] = int(line[1])


    def get_file_set(self, index):
        return self.file[index]

    def __len__(self):
        return len(self.imgs)



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

        dataSet = FlameSet('./data/input/COVIDClassification')
        train_loader = torch.utils.data.DataLoader(
            dataset=dataSet, batch_size=args.BATCH, shuffle=True
        )
        # print(dataSet[1]["data"][:3, ::].shape)
        crossentropy = nn.CrossEntropyLoss()
        net = Net()
        optim = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim)
        for epoch in range(args.EPOCHS):
            net.train()
            train_acc = 0.0
            train_loss = 0.0

            for batch_idx, (data, label) in enumerate(train_loader):
                logits = net.forward(data)
                loss = crossentropy(logits, torch.tensor(label))
                optim.zero_grad()
                loss.backward()
                optim.step()

                _, prediction = torch.max(logits.data, 1)

                train_loss += loss.item()
                train_acc += prediction.eq(label.data).sum()

                print(prediction)
                print(prediction.eq(label.data).sum())

                if batch_idx % 2 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.item()))

            # scheduler.step(train_loss, epoch)


if __name__ == '__main__':
    main = Main()
    main.download_data()
    main.deal_with_data()
    main.train()