import torch
from util.dataloader import DataLoader
from models.FreqPred3 import Freqpred
import numpy as np

''' define cpu or gpu '''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('>> Now we are using device:', device)


''' 读取数据 '''
# DataLoader是一个类
data = DataLoader(data_dir='data/new_rf_data.mat',  # .mat数据的文件路径
                  bin_size=2048,  # 预设划分的bin的个数
                  sample_rate=4e7)  # 采样频率
                  
# data = DataLoader(data_dir='data/original_data.npy',  # .mat数据的文件路径
#                   bin_size=2048,  # 预设划分的bin的个数
#                   sample_rate=2e7,using_group10_data=True,data_dir2 = 'data/original_data.npy')  # 采样频率
# # data.plot_data_raw(0)
# print(np.max(data.data_stft),np.min(data.data_stft),np.mean(data.data_stft))
# # ''' 数据预处理 '''
data.preprocess()
# print(np.max(data.data_pro),np.min(data.data_pro),np.mean(data.data_pro),np.std(data.data_pro))

data2 = DataLoader(data_dir='data/original_data.npy',  # .mat数据的文件路径
                  bin_size=2048,  # 预设划分的bin的个数
                  sample_rate=2e7,using_group10_data=True,data_dir2 = 'data/original_data.npy')  # 采样频率
data2.preprocess()
# print(np.max(data2.data_pro),np.min(data2.data_pro),np.mean(data2.data_pro),np.std(data2.data_pro))

ratio = data.merge_data(data2.data_pro)
# print(np.max(data.data_pro),np.min(data.data_pro),np.mean(data.data_pro),np.std(data.data_pro))
# print(ratio,data.data_pro.shape)
# p

# ''' 模型的基础设置 '''
# 期望由in_len个序列预测out_len个序列
# 只是规定了我们将从多少时间预测未来多少时间
# in_len, out_len的设置会影响数据集划分
model = Freqpred(in_len=100,
                 out_len=800,
                 device=device)
#
# ''' 划分训练集和测试集 '''
model.split_dataset(dataset=data.data_pro,  # 给模型输入数据
                    data_ratio=1-ratio)  # 划分比例
#
# ''' 模型网络初始化 '''
model.build(batch_size=40,
            lr=1e-3,
            num_layer=1)
#
# ''' 开始训练 '''
model.train(epochs=20,  # total epochs
            print_interval=1,  # 每隔多少epoch打印一次loss
            print_loss=True)  # 是否打印loss变化图
#
# ''' 测试结果 '''
model.test()


