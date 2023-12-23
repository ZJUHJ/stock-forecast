import os
import torch
import random
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from utils.model import build_model
from torch.utils.data import DataLoader
from utils.dataset import StockDatasetv3
from torch.utils.tensorboard import SummaryWriter
from utils.tools import AverageMeter, calculate_binary_classification_metrics


def train_one_iteration(model, dataset, optimizer, criterion, device):
    model.train()
    feature, label = dataset.get_train_batch()
    feature = feature.to(device)
    label = label.to(device)
    optimizer.zero_grad()
    output_logits = model(feature, label).squeeze(-1)
    label = label[:, -1]
    loss = criterion(output_logits, label)
    loss.backward()
    optimizer.step()

    return loss

def test(model, dataset, criterion, device):
    model.eval()
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
                calculate_binary_classification_metrics(output_logits, label, 0.5)
            tp_count += tp
            tn_count += tn
            fp_count += fp
            fn_count += fn
    acc = (tp_count + tn_count) / (tp_count + tn_count + fp_count + fn_count)
    recall = tp_count / (tp_count + fn_count)
    return acc, recall, loss_meter.get_avg()

def mainworker():
    #有问题：市销率
    factors = ['周开盘价','周收盘价','周最高价','周最低价','周涨跌幅','市盈率','市净率','市现率','ln_市值','Beta','波动率','换手率',\
            '3日均价','5日均价','7日均价','10日均价']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = StockDatasetv3('./data/new.csv', factors, split_ratio=0.8, split_type='date', max_window_size=10)
    dataset.gen_test_data(window_size=5)
    model = build_model(input_size=len(factors)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    dir_path = os.path.join('./log', current_time)
    os.makedirs(dir_path, exist_ok=True)
    writer = SummaryWriter(dir_path)
    
    train_loss_meter = AverageMeter('train loss')
    best_acc = 0.0
    best_recall = 0.0
    best_loss = 10000.0
    max_iteration_num = 100000
    for iter in tqdm(range(max_iteration_num)):
        loss = train_one_iteration(model, dataset, optimizer, criterion, device)
        train_loss_meter.update(loss.item())
        if (iter + 1) % 200 == 0:
            acc, recall, val_loss = test(model, dataset, criterion, device)
            writer.add_scalar('Loss/train', train_loss_meter.get_avg(), iter)
            writer.add_scalar('Loss/test', val_loss, iter)
            writer.add_scalar('Acc/test', acc, iter)
            writer.add_scalar('Recall/test', recall, iter)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), os.path.join(dir_path, 'best_loss.pth'))
                print(f'best loss model saved at iter {iter+1}, loss: {best_loss:.4f}')
            if acc > best_acc:  
                best_acc = acc
                torch.save(model.state_dict(), os.path.join(dir_path, 'best_acc.pth'))
                print(f'best acc model saved at iter {iter+1}, acc: {best_acc:.4f}')
            if recall > best_recall: 
                best_recall = recall
                torch.save(model.state_dict(), os.path.join(dir_path, 'best_recall.pth'))
                print(f'best recall model saved at iter {iter+1}, recall: {best_recall:.4f}')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    set_seed(331)
    mainworker()