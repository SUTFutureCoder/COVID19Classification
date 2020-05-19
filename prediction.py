# -*- coding: utf-8 -*
from flyai.framework import FlyAI
from path import MODEL_PATH

from PIL import Image
from main import train_transforms, DEVICE
import torch
import torchvision
import torch.utils.data as data

class SingleImgSet(data.Dataset):
    def __init__(self, root, transforms=None):
        self.transforms = transforms
        self.img_path = root

    def __getitem__(self, index):
        img_path = self.img_path
        img = Image.open(img_path).convert('RGB')
        return train_transforms(img)

    def __len__(self):
        return 1

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