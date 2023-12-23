import torch


class AverageMeter():
    def __init__(self, prefix=''):
        self.set_zero()
        self.prefix = prefix
    
    def update(self, v, n=1):
        self.total += n*v
        self.count += n
    
    def set_zero(self):
        self.total = 0.0
        self.count = 0
    
    def display(self):
        print(f'{self.prefix}: {self.total/self.count:.4f}')
    
    def get_avg(self):
        return self.total/self.count

def calculate_binary_classification_metrics(pred, label, threshold):
    pred = pred.sigmoid()
    binary_pred = (pred > threshold)
    tp_mask = (binary_pred == 1) & (label == 1)
    tn_mask = (binary_pred == 0) & (label == 0)
    fp_mask = (binary_pred == 1) & (label == 0)
    fn_mask = (binary_pred == 0) & (label == 1)
    return tp_mask.sum(), tn_mask.sum(), fp_mask.sum(), fn_mask.sum()

def save_model(model, path):
    torch.save(model.state_dict(), path)