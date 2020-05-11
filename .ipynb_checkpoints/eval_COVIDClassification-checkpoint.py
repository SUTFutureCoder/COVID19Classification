"""
线上评估文件
"""
import json
import os
import random
import sys
import urllib
from urllib import parse
import pandas as pd
import requests
import yaml
from flyai.framework import FlyAI
from flyai.processor.download import data_download
import numpy as np

class JsonHelper:
    def __init__(self, path=os.path.join(os.curdir, 'app.json')):
        self.path = path
        with open(path, 'r', encoding='utf-8') as f:
            try:
                self.data = json.loads(f.read())
            except json.decoder.JSONDecodeError:
                self.data = {}
                print("项目中的app.json文件格式错误，请检查后再运行")

    def get_train_class(self):
        if "project" in self.data:
            prediction = self.data["project"]['train']['file']
            if ".py" in prediction:
                prediction = os.path.splitext(prediction)[0]
            import importlib
            prediction = importlib.import_module(prediction)
            models = dir(prediction)
            for model in models:
                if not model.startswith("__") and model is not "FlyAI":
                    clazz = getattr(prediction, model)
                    if isinstance(clazz, type) and issubclass(getattr(prediction, model), FlyAI):
                        return clazz
        print("评估系统找不到对应的训练类，请在main.py文件中添加对应的训练方法")
        return self.data

    def get_prediction_class(self):
        if "project" in self.data:
            prediction = self.data['project']['prediction']['file']
            if ".py" in prediction:
                prediction = os.path.splitext(prediction)[0]
            import importlib
            prediction = importlib.import_module(prediction)
            models = dir(prediction)
            for model in models:
                if not model.startswith("__") and model is not "FlyAI":
                    clazz = getattr(prediction, model)
                    if isinstance(clazz, type) and issubclass(getattr(prediction, model), FlyAI):
                        return clazz
        print("评估系统找不到对应的评估类，请在prediction.py文件中添加对应的评估方法")
        return self.data

    def get_projcet_id(self):
        if 'project':
            return self.data['project']['id']
        return {}

    def data_config(self):
        if 'data' in self.data:
            return self.data['data']
        return self.data

    def server_config(self):
        if 'server' in self.data:
            return self.data['servers']
        return self.data

    def model_config(self):
        if 'model' in self.data:
            return self.data['model']
        return self.data

    def get_input_names(self):
        if 'model' in self.data:
            config = self.data['model']
            input = config['input']
            names = []
            for columns in input['columns']:
                names.append(columns['name'])
            return names
        return self.data

    def get_input_shape(self):
        config = self.data['model']
        input = config['input']
        return input['shape']

    def get_output_names(self):
        if 'model' in self.data:
            config = self.data['model']
            input = config['output']
            names = []
            for columns in input['columns']:
                names.append(columns['name'])
            return names
        return self.data

    def get_output_shape(self):
        config = self.data['model']
        input = config['output']
        return input['shape']

    def get_data_id(self):
        if 'data' in self.data:
            return self.data['data']['id']
        return self.data

    def get_servers(self):
        if 'servers' in self.data:
            return self.data['servers']
        return self.data

    def processor(self):
        if 'model' in self.data:
            processor = dict()
            processor['processor'] = self.data['model']['processor']
            processor['input_x'] = self.data['model']['input_x']
            processor['input_y'] = self.data['model']['input_y']
            try:
                output_x = self.data['model']['output_x']
            except:
                output_x = self.data['model']['input_x']
            processor['output_x'] = output_x
            processor['output_y'] = self.data['model']['output_y']
            return processor
        return self.data

    def get_data_source(self):
        if "data" in self.data:
            return self.data['data']['source']
        return self.data

    def set_data_source(self, data):
        if "data" in self.data:
            self.data['data']['source'] = data
            with open(self.path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(self.data, ensure_ascii=False, indent=2, separators=(', ', ':')))
                return self.data

    def to_yaml(self):
        file = os.path.join(os.curdir, 'app.yaml')
        stream = open(file, 'w')
        yaml_data = self.data
        yaml_data['algorithm'] = self.data['project']['algorithm']
        yaml_data['language'] = self.data['project']['language']
        yaml_data['framework'] = self.data['project']['framework']
        yaml_data['data']['id'] = self.data['project']['id']
        yaml.safe_dump(self.data, stream=stream, default_flow_style=False)


try:
    version = 1
    from model import Model
    from flyai.utils.y import Yaml

    helper = Yaml()
except:

    helper = JsonHelper()
    version = 2


def get_json(url, is_log=False):
    try:
        response = requests.get(url=url)
        if is_log:
            print("server code ", response, url)
        return response.json()
    except:
        return None


'''
不需要修改

'''

if "https" in sys.argv[1]:
    data_id = sys.argv[1].split('/')[4]
else:
    data_id = sys.argv[1]
data_path = get_json("https://www.flyai.com/get_evaluate_command2?data_id=" + data_id)


def read(path):
    try:
        return pd.read_csv(path)
    except OSError:
        return pd.read_csv(path, engine="python")


def get_evaluate_data(path):
    DATA_PATH = os.path.join(os.curdir, 'data', 'input')
    evaluate_path = data_download(path, DATA_PATH, is_print=False)
    evaluate = read(evaluate_path + "/validation.csv")
    os.remove(evaluate_path + "/validation.csv")
    return get_data(evaluate)


def get_data(data):
    x_names = helper.get_input_names()
    y_names = helper.get_output_names()
    x_data = data[x_names]
    y_data = data[y_names]
    x_data = x_data.to_dict(orient='records')
    y_data = y_data.to_dict(orient='records')
    return x_data, y_data


x_test, y_test = get_evaluate_data(urllib.parse.unquote(data_path['command']))

randnum = random.randint(0, 100)
random.seed(randnum)
random.shuffle(x_test)

random.seed(randnum)
random.shuffle(y_test)
eval = 0
if version == 1:
    from flyai.dataset import Dataset

    dataset = Dataset()
    model = Model(dataset)
    labels = model.predict_all(x_test)
elif version == 2:
    labels = []
    prediction = helper.get_prediction_class()
    prediction = prediction()
    prediction.load_model()
    s = 0
    for index, x in enumerate(x_test):
        s += 1
        y = prediction.predict(**x)
        labels.append(y)
        if index % 100 == 0:
            print('%d/%d predict done!!!' % (s, len(x_test)))
'''
if不需要修改

'''
if len(y_test) != len(labels):
    result = dict()
    result['score'] = 0
    result['label'] = "评估违规"
    result['info'] = ""
    print(json.dumps(result))
else:
    '''
    在下面实现不同的评估算法
    '''

    y_true = [i['label'] for i in y_test]
    y_scores = [i['label'] for i in labels]
    
    # acc 准确率
    all_len = len(y_true)
    acc_num = 0.0
    for i in range(all_len):
        if y_true[i] == y_scores[i]:
            acc_num += 1

    eval = acc_num / all_len

    result = dict()
    result['score'] = round(eval * 100, 2)
    result['label'] = "The score is acc"
    result['info'] = ""
    print(json.dumps(result))