# -*- coding: utf-8 -*-
import argparse
import os

from flyai.data_helper import DataHelper
from flyai.framework import FlyAI

from path import MODEL_PATH

import csv
from PIL import Image
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

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
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()

# 定义超参数
IMG_SIZE = 256
INPUT_SIZE = 244
BATCH_SIZE = args.BATCH
EPOCHS = args.EPOCHS
BASE_LR = 0.01
CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transforms = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomCrop(INPUT_SIZE),
    transforms.ToTensor(),
    normalize
])

class ConvDataSet(data.Dataset):
    def __init__(self, root, transforms=None):
        img_root = root + "/image"
        train_root = root + "/train.csv"
        self.img_path = [os.path.join(img_root, k) for k in os.listdir(img_root)]
        self.img_key = os.listdir(img_root)
        self.transforms = transforms
        self.labels = self.__getlabel__(train_root)

    def __getitem__(self, index):
        img_path = self.img_path[index]
        img = Image.open(img_path).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        return img, self.labels[self.img_key[index]]

    def __getlabel__(self, train_file):
        results = {}
        with open(train_file, 'r') as trainCsv:
            lines = csv.reader(trainCsv)
            for line in lines:
                if line[1] == 'label':
                    continue
                key = line[0].split('/')[1]
                results[key] = int(line[1])
        return results

    def __len__(self):
        return len(self.img_key)

def ResNet152():
    model = torchvision.models.resnet152(pretrained=True)

    # freeze
    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(DEVICE)
    return model


def train(model, dataloader_dict, optimizer, criterion):
    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    acc_his = []

    for epoch in range(EPOCHS):
        print('EPOCH {} / {}'.format(epoch, EPOCHS - 1))
        print('-'*10)

        optim.lr_scheduler.StepLR(optimizer, 15, 0.01)

        for phase in ['train', 'eval']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_correct = 0

            for inputs, labels in dataloader_dict[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    out = model(inputs)
                    loss = criterion(out, labels)

                _, pred = torch.max(out, 1)

                running_loss += loss.item() / inputs.size(0)
                running_correct += torch.sum(pred == labels.data)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            epoch_loss = running_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = running_correct.double() / len(dataloader_dict[phase].dataset)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                epoch_acc = best_acc
                best_model = copy.deepcopy(model.state_dict())
            if phase == 'val':
                acc_his.append(epoch_acc)

    model.load_state_dict(best_model)
    return model, acc_his


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
        dataset = ConvDataSet("./data/input/COVIDClassification", transforms=train_transforms)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size])

        dataloader_dict = {
            'train': data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True),
            'eval': data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
        }

        model = ResNet152()
        if not CUDA:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss().cuda(DEVICE)
        optimizer = optim.SGD(model.parameters(), BASE_LR, momentum=0.9, weight_decay=0.0001)

        model, acc_his = train(model, dataloader_dict, optimizer, criterion)
        print(acc_his)
        torch.save(model.state_dict(), MODEL_PATH + '/mdl.pkl')
        pass


if __name__ == '__main__':
    main = Main()
    main.download_data()
    main.train()
