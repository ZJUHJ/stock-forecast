from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

# 加载 TensorBoard 日志文件
event_acc = EventAccumulator('/Users/dqai/Desktop/DataDig/log/20231223_011659')
event_acc.Reload()

# 获取损失数据
loss_data = event_acc.Scalars('Recall/test')

# 提取数据
steps = [entry.step for entry in loss_data]
values = [entry.value for entry in loss_data]

# 使用 matplotlib 绘制图表
plt.figure()
plt.plot(steps, values)
plt.xlabel('Steps')
plt.ylabel('Recall')

# 保存图表
plt.savefig('./resources/figs/recall_test.svg', format='svg')