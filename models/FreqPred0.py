import torch
from torch import nn
import numpy as np
from tqdm import tqdm,trange
from util.plot_tools import Plot_signal, Plot_stft
import matplotlib.pyplot as plt
import time
# N是训练/测试数据的条数
# I是输入序列长度
# O是输出序列长度
# B是频率轴上的长度(即bin_size)
# C是数据条数(如每个mat由10条数据)
# BS是batch_size


class lstm(nn.Module): #修改model的同学看这里
    def __init__(self,
                 input_size=256,
                 hidden_size=256,
                 output_size=256,
                 num_layer=2,
                 input_len=50,
                 output_len = 10):
        super(lstm, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Sequential(nn.Linear(hidden_size,hidden_size))
        self.layer3 = nn.Sequential(nn.Linear(hidden_size,output_size))
        # self.layer2 = nn.Sequential(nn.Linear(hidden_size,16))
        # self.layer3 = nn.Sequential(nn.Linear(16*input_len,output_size))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.output_len = output_len
        # self.batch_norm = nn.BatchNorm1d()

    def forward(self, x):
        # x = self.layer0(x)
        x, (h, c) = self.layer1(x)  # 输入：50个向量 输出：x：50个向量
        # last_output = x[-1:]
        last_output = x
        # print(last_output.shape,"*")
        # last_output =  last_output
        output = self.layer2(last_output)
        output = self.relu(output)
        # print(output.shape)
        # output = output.permute(1,0,2)
        # output = output.reshape((output.shape[0],1,-1))
        output = self.layer3(output)
        # output = self.sigmoid(output)


        # print(output.shape,"*")
        output = torch.mean(output,dim=0)
        # print(output.shape,"*")
        
        # print(output.shape)
        # for i in range(self.output_len): # 每次的预测结果作为下一次输入
        #     y, (h, c) = self.layer1(last_output,(h,c))
        #     last_output = y
        #     if i==0:
        #         output = y
        #     else:
        #         output = torch.cat((output, y), 0)
        # output = self.layer2(output)
        # output = self.sigmoid(output)
        return output


class Freqpred:
    def __init__(self, in_len, out_len, device):
        self.in_len = in_len  # input_seq_len
        self.out_len = out_len  # output_seq_len
        self.device = device
        self.bin_size = None
        self.train_x, self.test_x = None, None  # (N, I, B), 训练集的输入部分，测试集的输入部分
        self.train_y, self.test_y = None, None  # (N, O, B), 训练集的输出部分，测试集的输出部分
        self.model = None  # 网络模型
        self.optimizer = None  # 优化器
        self.criterion = None  # loss func
        self.batch_size = None
        self.test_batch_size = 1 # modified jmy # when testing the batch size
        self.last_time = 0
        self.lr = None
        self.history_loss = []  # 保存每次训练每轮的平均loss
        self.total_loss = []

    def split_dataset(self, dataset, data_ratio=0.7):  # (C, B, T)
        # 数据集划分
        print("-----<Dataset Split>------")
        self.bin_size = dataset.shape[1]  # B
        x, y = self.create_dataset(dataset.transpose((0, 2, 1)))  # input=(C, T, B), output=(C, T, I/O, B), (C, T, O, B)
        data_size = x.shape[0]*x.shape[1]  # C
        # 划分训练集和测试集，70% 作为训练集
        train_size = int(data_size * data_ratio)
        self.train_x = x.reshape((-1, self.in_len, self.bin_size))[:train_size]  # (N, I, B)
        self.train_y = y.reshape((-1, 1, self.bin_size))[:train_size]  # (N, 1, B)
        self.test_x = x.reshape((-1, self.in_len, self.bin_size))[train_size:]  # (N, I, B)
        self.test_y = y.reshape((-1, 1, self.bin_size))[train_size:]  # (N, 1, B)
        print(f">> train_set: {self.train_x.shape[0]} sequences, test_set: {self.test_x.shape[0]} sequences. ")

    # def create_dataset(self, dataset):  # (C, T, B)
    #     C, T, B = dataset.shape
    #     x = np.array([dataset[:, i:(i + self.in_len)]
    #              for i in range(T - self.in_len - self.out_len + 1)])  # (new_T, C, O, B)
    #     y = np.array([dataset[:, i + self.in_len:i + self.in_len + self.out_len]
    #              for i in range(T - self.in_len - self.out_len + 1)])  # (new_T, C, I, B)
    #     return x.transpose((1, 0, 2, 3)), y.transpose((1, 0, 2, 3))  # (C, T, I/O, B)

    def create_dataset(self, dataset):  # jmy # create data with no overlap
        C, T, B = dataset.shape
        x = np.array([dataset[:, i*self.in_len:(i +1 )*self.in_len]
                 for i in range((T - self.in_len - self.out_len + 1)//self.in_len)])  # (new_T, C, O, B)
        y = np.array([dataset[:,(i +1 )*self.in_len:(i +1)*self.in_len + self.out_len]
                 for i in range((T - self.in_len - self.out_len + 1)//self.in_len)])  # (new_T, C, I, B)
        # print("&",y.shape)
        y = np.mean(y,axis=2)
        # print("&",y.shape)
        # p
        return x.transpose((1, 0, 2, 3)), y#.transpose((1, 0, 2, 3))  # (C, T, I/O, B)

    def build(self, batch_size=40, lr=1e-3, num_layer=1):
        # 网络初始化
        self.batch_size = batch_size
        self.lr = lr
        # 设置模型，修改模型结构的同学改这部分
        self.model = lstm(input_size=self.bin_size,
                          hidden_size=self.bin_size,
                          output_size=self.bin_size,
                          num_layer=num_layer,
                          input_len=self.in_len,
                          output_len=self.out_len
                          ).to(self.device)
        # self.criterion = nn.MSELoss().to(self.device)
        self.criterion = nn.L1Loss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, epochs=1, print_interval=1, print_loss=True):
        print("-----<Training>------")
        self.model = self.model.train()
        self.history_loss = []
        with trange(epochs) as t:
            for e in t:
                loss_list = []
                range_list = np.arange(0, self.train_x.shape[0], self.batch_size)
                np.random.shuffle(range_list)
                mini_batch_x = [self.train_x[k:k + self.batch_size] for k in range_list]
                mini_batch_y = [self.train_y[k:k + self.batch_size] for k in range_list]
    
                for i in range(len(mini_batch_x)):
                    # (BS, I/O, B) -> (I/O, BS, B)
                    x = np.array(mini_batch_x[i]).transpose((1, 0, 2))  # 为了适应LSTM参数：第二个维度是batch_size
                    y = np.array(mini_batch_y[i]).transpose((1, 0, 2))
                    x = torch.from_numpy(x).to(self.device)  # (I, BS, B)
                    y = torch.from_numpy(y).to(self.device)  # (O, BS, B)
    
                    # 前向传播 truelabel设计？
                    out = self.model(x)
                    loss = self.criterion(out, y)
                    loss_list.append(loss.item())
                    # 反向传播
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
    
                mean_loss = np.array(loss_list).mean()
                self.history_loss.append(mean_loss)
                if (e + 1) % print_interval == 0:  # 每 print_interval 次输出一次结果
                    t.set_description('Epoch: {}'.format(e+1))
                    t.set_postfix(Loss = mean_loss)
                    # print('Epoch: {}, Loss: {}'.format(e + 1, mean_loss))

        if print_loss:
            Plot_signal(data=np.array(self.history_loss),
                        title='History Loss',
                        xlabel='Epoch',
                        ylabel='Loss')

    def test(self):
        print("-----<Testing>------")
        self.model = self.model.eval()
        x = self.test_x.transpose((1, 0, 2))
        print(x.shape)
        mini_batch_x = [x[:, k*self.test_batch_size:(k+1)*self.test_batch_size]\
                        for k in range(x.shape[1]//self.test_batch_size)]
        if x.shape[1] % self.test_batch_size != 0:
            mini_batch_x.append(x[:, x.shape[1]//self.test_batch_size*self.test_batch_size:])

        y_pred = None
        last_x = None
        last_y = None
        for i in tqdm(range(len(mini_batch_x))):
            # (BS, I, B) -> (I, BS, B)
            x = np.array(mini_batch_x[i])  # 为了适应LSTM参数：第二个维度是batch_size
            # jmy# v2: 不用新数据,一直滚着预测
            # if i>0:
            #     print(last_x.shape,last_y.transpose(1, 0, 2).shape)
            #     x = np.concatenate([last_x,last_y.transpose(1, 0, 2) ],axis=0)
            #     x = x[last_y.shape[1]:,:,:]
            #     print(x.shape,"*")
            last_x = x
            x = torch.from_numpy(x).to(self.device)
            # print("x.shape",x.shape)
            y = self.model(x).cpu().data.numpy()
            # print(y.shape)
            # y = y.transpose(1, 0, 2)  # 测试集的预测结果 (BS, O, B)
            
            ## modified jmy
            # v1: 实时显示histogram
            predict_acc_each_bin = y
            y_gt_minibatch = self.test_y[i:i+1,:]
            
            # if i>0:
            self.dynamic_plot(predict_acc_each_bin,y_gt_minibatch)
            last_y = y_gt_minibatch
            

            
            if i == 0:
                y_pred = y
            else:
                y_pred = np.concatenate([y_pred, y], axis=0)
                
        #阈值判断
        # y_pred = y_pred>0.31

        y_gt = self.test_y  # (N, O, B)
        acc = self.compute_acc(y_pred, y_gt)
        print(">> Acc:", acc)
        print("loss:",np.mean(self.total_loss))
        # yyy = y_pred.transpose(1, 0, 2).reshape((-1, 256))
        Plot_stft(data=y_pred[:, 0, :].T)
        Plot_stft(data=y_gt[:, 0, :].T)

    def compute_acc(self, pred, gt):
        N, O, B = gt.shape  # (N, O, B)
        return np.sum((pred == gt))/(N * O * B)

    def dynamic_plot(self,pred,gt):# jmy
        # print("#",pred.shape,gt.shape)
        print("time:",time.time()-self.last_time)
        self.last_time = time.time()
        pred = pred.reshape((-1))
        gt = gt.reshape((-1))
        self.total_loss.append( np.sum(np.abs(pred-gt)))
        # pred = np.sum(pred,axis=1)
        # pred = np.sum(pred,axis=0)
        # now pred shape is (256,)
        # gt = np.sum(gt,axis=1)
        # gt = np.sum(gt,axis=0)
        plt.clf()
        plt.plot(pred,label="prediction")
        plt.plot(gt,label="ground truth")
        plt.ylim((0,1))
        plt.legend()
        plt.draw()
        plt.pause(0.0001)
