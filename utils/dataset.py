from typing import Any
import torch
import random
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from torch.utils.data import Dataset


class StockDataset(Dataset):
    def __init__(self, path, factors, window_size=5, split_type='date', is_train=True):
        self.split_type = split_type
        self.factors = factors
        self.window_size = window_size
        self.is_train = is_train
        self.codes, self.data = self.load_data(path, factors)
    
    def load_data(self, path, factors):
        data = pd.read_csv(path, encoding='gbk', dtype={0: str, 1: str})
        data = data.dropna(how='all')
        data = data[factors]
        codes = []
        buffer = {}
        for index, row in data.iterrows():
            if row['代码'] not in codes:
                codes.append(row['代码'])
                buffer[row['代码']] = {}
            buffer[row['代码']][row['日期']] = []
            for factor in factors[2:]:
                num = row[factor]
                if type(num) == str:
                    num = float(eval(num))
                elif type(num) != str:
                    num = float(num)
                buffer[row['代码']][row['日期']].append(row[factor])

        for code in codes:
            try:
                buffer[code] = sorted(buffer[code].items(), key=lambda item: datetime.strptime(item[0], '%Y/%m/%d'))
            except:
                buffer[code] = sorted(buffer[code].items(), key=lambda item: datetime.strptime(item[0], '%Y-%m-%d'))

        return codes, buffer

    def __len__(self):
        return len(self.codes)
        
    def __getitem__(self, idx):
        code = self.codes[idx]
        code_data = self.data[code]
        data = []
        if self.is_train:
            index = random.randint(0, len(code_data) - 1)
            for i in range(index+1-self.window_size, index+1):
                if i < 0:
                    data.append(code_data[0][1])
                elif i > len(code_data) - 1:
                    data.append(code_data[len(code_data)][1])
                else:
                    data.append(code_data[i][1])
            label = int(data[-1][-1])
            data = torch.tensor(data, dtype=torch.float32)[:,:-1]
        else:
            pass
        return data, label


class StockDatasetv2(Dataset):
    def __init__(self, path, factors, split_ratio=0.5, normalize=True, window_size=5, split_type='date', is_train=True):
        self.split_type = split_type
        self.factors = factors
        self.window_size = window_size
        self.is_train = is_train
        self.splt_ratio = split_ratio
        self.normalize = normalize
        self.data = self.load_data(path, factors)
        self.keys = list(self.data.keys())
    
    def load_data(self, path, factors):
        data = pd.read_csv(path, encoding='gbk', dtype={0: str, 1: str})
        data = data.dropna(how='all')
        buffer = {}
        for index, row in data.iterrows():
            code = row['代码']
            date = row['日期']
            features = []
            for factor in factors:
                num = row[factor]
                if type(num) == str:
                    if ',' in num:
                        num = num.replace(',', '')
                    num = float(num)
                features.append(num)
            label = row['lable_下周涨跌']
            if code not in buffer:
                buffer[code] = {'features': [], 'labels': []}
            buffer[code]['features'].append(features)
            buffer[code]['labels'].append(label)
        for key in buffer.keys():
            features = torch.tensor(buffer[key]['features'], dtype=torch.float32) # (104,17)
            if self.normalize:
                mean = features.mean(dim=0, keepdim=True) # (1, 17)
                var = features.var(dim=0, keepdim=True) # (1, 17)
                features = (features - mean) / var # (104, 17)
            buffer[key]['features'] = features
        return buffer
        

    def __len__(self):
        return len(self.keys)
        
    def __getitem__(self, idx):
        code = self.keys[idx]
        code_features = self.data[code]['features']
        code_labels = self.data[code]['labels']
        window_size = self.window_size
        if self.is_train:
            idx = random.randint(0, int(len(code_labels)*self.splt_ratio) - window_size - 1)
        else:
            pass
        features = code_features[idx:idx+window_size]
        labels = code_labels[idx:idx+window_size]
        labels = torch.Tensor(labels).float()
        return features, labels


class StockDatasetv3():
    def __init__(self, path, factors, split_ratio=0.5, normalize=True, max_window_size=5, split_type='date'):
        self.split_type = split_type
        self.factors = factors
        self.max_window_size = max_window_size
        self.split_ratio = split_ratio
        self.normalize = normalize
        self.data = self.load_data(path, factors)
        self.codes = list(self.data.keys())
        self.split_code()
        self.gen_test_data()
    
    def load_data(self, path, factors):
        data = pd.read_csv(path, encoding='gbk', dtype={0: str, 1: str})
        data = data.dropna(how='all')
        buffer = {}
        for index, row in data.iterrows():
            code = row['代码']
            date = row['日期']
            features = []
            for factor in factors:
                num = row[factor]
                if type(num) == str:
                    if ',' in num:
                        num = num.replace(',', '')
                    num = float(num)
                features.append(num)
            label = row['lable_下周涨跌']
            if code not in buffer:
                buffer[code] = {'features': [], 'labels': []}
            buffer[code]['features'].append(features)
            buffer[code]['labels'].append(label)
        for key in buffer.keys():
            features = torch.tensor(buffer[key]['features'], dtype=torch.float32) # (104,17)
            if self.normalize:
                mean = features.mean(dim=0, keepdim=True) # (1, 17)
                var = features.var(dim=0, keepdim=True) # (1, 17)
                features = (features - mean) / var # (104, 17)
            buffer[key]['features'] = features
        return buffer
    
    def split_code(self):
        if self.split_type == 'code': # 按股票代码划分
            self.train_codes = random.sample(self.codes, int(len(self.codes)*self.split_ratio))
            self.val_codes   = []
            for code in self.codes:
                if code not in self.train_codes:
                    self.val_codes.append(code)
        else: # 按日期划分或随机划分
            self.train_codes = self.codes
            self.val_codes = self.codes
            
    def get_train_item(self, window_size=5):
        if self.split_type == 'random':
            pass
        else:
            code = random.choice(self.train_codes)
            code_features = self.data[code]['features']
            code_labels = self.data[code]['labels']
            length = len(code_labels)
            if self.split_type == 'date':
                st_idx = random.randint(0, int(length*self.split_ratio)-window_size)
            elif self.split_type == 'code':
                st_idx = random.randint(0, length-window_size-1)
            features = code_features[st_idx: st_idx+window_size]
            labels = code_labels[st_idx: st_idx+window_size]

        return features, labels
    
    def get_train_batch(self, batch_size=32):
        batch_features = []
        batch_labels   = []
        window_size = random.randint(1, self.max_window_size) # 随机生成窗口大小
        for i in range(batch_size):
            features, labels = self.get_train_item(window_size)
            batch_features.append(features)
            batch_labels.append(labels)
        batch_features = torch.stack(batch_features).float()
        batch_labels = torch.Tensor(batch_labels).float()
        return batch_features, batch_labels
    
    def gen_test_data(self, window_size=5):
        self.test_features, self.test_labels = [], []
        if self.split_type == 'random':
            for code in self.val_codes:
                n = len(self.data[code]['labels'])
                train_ids = random.sample(list(range(n)), int(n*self.split_ratio))
        else:
            for code in self.val_codes:
                code_features = self.data[code]['features']
                code_labels = self.data[code]['labels']
                if self.split_type == 'date':
                    st_idx = int(len(code_labels)*self.split_ratio)
                elif self.split_type == 'code':
                    st_idx = 0
                for i in range(st_idx, len(code_labels)-window_size-1, window_size):
                    features = code_features[i:i+window_size]
                    labels = code_labels[i:i+window_size]
                    self.test_features.append(features)
                    self.test_labels.append(labels)
    
    def get_test_batch(self, batch_size=32):
        n = len(self.test_features)
        features, labels = [], []
        for i in range(0, n, batch_size):
            batch_features = self.test_features[i:i+batch_size]
            batch_labels = self.test_labels[i:i+batch_size]
            batch_features = torch.stack(batch_features).float()
            batch_labels = torch.Tensor(batch_labels).float()
            features.append(batch_features)
            labels.append(batch_labels)
        return features, labels


if __name__ == '__main__':
    factors = ['周开盘价','周收盘价','周最高价','周最低价','周涨跌幅','市盈率','市净率','市现率','ln_市值','Beta','波动率','换手率',\
            '3日均价','5日均价','7日均价','10日均价']
    window_size = 1
    dataset = StockDatasetv3('./data/new.csv', factors, split_ratio=0.8, split_type='random', max_window_size=window_size)
    
    pos_count = 0
    total_count = 0
    for i in range(1000000):
        features, labels = dataset.get_train_batch()
        pos_count += labels.sum()
        total_count += labels.numel()
    print(pos_count/total_count) # date:0.4804, code: 0.4790
