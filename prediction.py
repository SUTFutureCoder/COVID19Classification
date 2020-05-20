# -*- coding: utf-8 -*
import matplotlib
from flyai.framework import FlyAI
from path import MODEL_PATH

from PIL import Image
from main import train_transforms, DEVICE
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import csv

class SingleImgSet(data.Dataset):
    def __init__(self, root, transforms=None):
        self.transforms = transforms
        self.img_path = root

    def __getitem__(self, index):
        img_path = self.img_path
        img = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(244),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(img)

    def __len__(self):
        return 1

class Pred_dataset(data.Dataset):
    def __init__(self, root):
        image_root = root + "/image2"
        train_root = root + "/train.csv"
        self.img_path = [os.path.join(image_root, k) for k in os.listdir(image_root)]
        self.img_key = os.listdir(image_root)
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(244),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.labels = self.__getlabel__(train_root)

    def __getitem__(self, index):
        img_path = self.img_path[index]
        img = Image.open(img_path).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        return img, (img_path, self.labels[self.img_key[index]])

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


# noinspection PyUnresolvedReferences,PyStatementEffect
class Prediction(FlyAI):
    def load_model(self):
        '''
        模型初始化，必须在此方法中加载模型
        '''
        self.model_state_dict = torch.load(MODEL_PATH + '/mdl.pkl')
        pass

    def predict(self, image_path):
        '''
        模型预测返回结果
        :param input:  评估传入样例 {"image_path":".\/data\/input\/COVIDClassification\/image\/0.png"}
        :return: 模型预测成功之后返回给系统样例 {"label":"0"}
        '''
        model = torchvision.models.resnet152(pretrained=False, num_classes=2)
        model.eval()
        model.load_state_dict(self.model_state_dict)

        img_set = SingleImgSet(image_path['image_path'])
        for input in data.DataLoader(img_set):
            input.to(DEVICE)
            out = model(input)
            _, pred = torch.max(out, 1)
            return {"label":str(pred.item())}


        return {"label":"0"}

    def predict_pics(self, image_path):
        model = torchvision.models.resnet152(pretrained=False, num_classes=2)
        model.eval()
        model.load_state_dict(self.model_state_dict)

        import matplotlib.pyplot as plt

        plt.figure()
        plt.ion()
        # plt.title('COVID19Classification')

        img_set = Pred_dataset(image_path)
        i = 1
        for input, labels in data.DataLoader(img_set):
            input.to(DEVICE)
            out = model(input)
            _, pred = torch.max(out, 1)
            # print(labels)
            # print(pred.item())

            plt.subplot(3, 3, i)
            i += 1
            if pred.item() == 0:
                plt.title("Uninfected")
            else:
                plt.title("Infected")
            plt.imshow(input.cpu()[0, -1])

        plt.show()
