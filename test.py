import os
import torch
import random
from tqdm import tqdm
import torch.nn as nn
from datetime import datetime
from utils.model import build_model
from torch.utils.data import DataLoader
from utils.dataset import StockDatasetv3
from utils.tools import AverageMeter, calculate_binary_classification_metrics


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # np.random.seed(seed)
    random.seed(seed)

def control_test(model, dataset, criterion, device, window_size=5, threshold=0.5):
    model.eval()
    dataset.gen_test_data(window_size=window_size)
    features, labels = dataset.get_test_batch()
    tp_count, tn_count, fp_count, fn_count = 0, 0, 0, 0
    loss_meter = AverageMeter('test loss')
    with torch.no_grad():
        for feature, label in zip(features, labels):
            feature = feature.to(device)
            label = label.to(device)
            output_logits = model(feature, label).squeeze(-1)
            label = label[:, -1]
            loss = criterion(output_logits, label)
            loss_meter.update(loss.item(), n=output_logits.shape[0])
            tp, tn, fp, fn =\
                calculate_binary_classification_metrics(output_logits, label, threshold)
            tp_count += tp
            tn_count += tn
            fp_count += fp
            fn_count += fn
    acc = ((tp_count + tn_count) / (tp_count + tn_count + fp_count + fn_count)).item()
    TPR = (tp_count / (tp_count + fn_count)).item()
    FPR = (fp_count / (fp_count + tn_count)).item()
    return acc

def main():
    factors = ['周开盘价','周收盘价','周最高价','周最低价','周涨跌幅','市盈率','市净率','市现率','ln_市值','Beta','波动率','换手率',\
            '3日均价','5日均价','7日均价','10日均价']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = StockDatasetv3('./data/new.csv', factors, split_ratio=0.8, split_type='date', max_window_size=10)
    model = build_model(input_size=len(factors)).to(device)
    model.load_state_dict(torch.load('./log/20231223_011659/best_acc.pth'))
    criterion = nn.BCEWithLogitsLoss()
    for w in range(1, 11):
        acc = control_test(model, dataset, criterion, device, window_size=w)
        print(f'window size: {w}, acc: {acc:.4f}')


if __name__ == '__main__':
    set_seed(331)
    main()