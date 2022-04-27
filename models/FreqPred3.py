import torch
from torch import nn
from tqdm import tqdm, trange
from util.plot_tools import Plot_signal, Plot_stft
import matplotlib.pyplot as plt
import os
from pathlib import Path
import datetime
import numpy as np
# N是训练/测试数据的条数
# I是输入序列长度
# O是输出序列长度
# B是频率轴上的长度(即bin_size)
# C是数据条数(如每个mat由10条数据)
# BS是batch_size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class lstm(nn.Module):  # 修改model的同学看这里
    def __init__(self,
                 input_size=256,
                 hidden_size=256,
                 output_size=256,
                 num_layer=2,
                 output_len=10):
        super(lstm, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_size,hidden_size),nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(hidden_size,input_size),nn.ReLU())
        self.layer1 = nn.LSTM(hidden_size, hidden_size, num_layer)
        self.layer2 = nn.Sequential(nn.Linear(hidden_size,hidden_size))
        self.layer3 = nn.Sequential(nn.Linear(hidden_size,output_size))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.output_len = output_len

    def forward(self, x):
        encode_x = self.encoder(x)
        decode_x = self.decoder(encode_x)
        x = encode_x.detach()

        x, (h, c) = self.layer1(x)  
        last_output = x
        output = self.layer2(last_output)
        output = self.relu(output)
        output = self.layer3(output)
        output = torch.mean(output, dim=0)

        return output, decode_x


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
        self.lr = None
        self.history_loss = []  # 保存每次训练每轮的平均loss
        self.test_loss = []
        self.base_dir = None

    def split_dataset(self, dataset, data_ratio):  # (C, B, T)
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

    def create_dataset(self, dataset):  # jmy # create data with no overlap
        C, T, B = dataset.shape
        x = np.array([dataset[:, i*self.in_len:(i+1)*self.in_len]
                 for i in range((T - self.in_len - self.out_len + 1)//self.in_len)])  # (new_T, C, O, B)
        y = np.array([dataset[:,(i+1)*self.in_len:(i+1)*self.in_len + self.out_len]
                 for i in range((T - self.in_len - self.out_len + 1)//self.in_len)])  # (new_T, C, I, B)
        y = np.mean(y, axis=2)
        return x.transpose((1, 0, 2, 3)), y  # .transpose((1, 0, 2, 3))  # (C, T, I/O, B)

    def build(self, batch_size, lr, num_layer, weight_name='', load_dir=''):
        # 网络初始化
        self.base_dir = Path(os.path.join('logs', 'model'))
        self.base_dir.mkdir(parents=True, exist_ok=True)
        if weight_name == '':
            weight_name = 'model_'+str(datetime.datetime.now().strftime('%y%m%d_%H%M'))
        self.save_dir = Path(os.path.join(self.base_dir, weight_name))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.lr = lr
        # 设置模型，修改模型结构的同学改这部分
        self.model = lstm(input_size=self.bin_size,
                          hidden_size=self.bin_size,
                          output_size=self.bin_size,
                          num_layer=num_layer,
                          output_len=self.out_len
                          ).to(self.device)
        if load_dir != '':
            weight = torch.load(os.path.join(self.base_dir, load_dir))
            self.model.load_state_dict(weight)
            print(">> Model_weight is loaded.")
        self.criterion = nn.L1Loss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.optimizer_encoder = torch.optim.Adam(self.model.parameters(), lr=self.lr*0.1)#jmy
        self.criterion_encoder = nn.MSELoss().to(self.device)

    def train(self, epochs, print_interval, save_interval=10, print_loss=True):
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
                    out, decoded_input = self.model(x)
                    loss = self.criterion(out, y[0]) +self.criterion_encoder(decoded_input,x.detach())*0.1#jmy
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
                if (e + 1) % save_interval == 0:
                    model_weight = self.model.state_dict()
                    savepath = os.path.join(self.save_dir, str(e)+'.pth')
                    torch.save(model_weight, savepath)

        if print_loss:
            Plot_signal(data=np.array(self.history_loss),
                        title='History Loss',
                        xlabel='Epoch',
                        ylabel='Loss')

    def test(self):
        print("-----<Testing>------")
        # self.model = self.model.eval()
        x = self.test_x.transpose((1, 0, 2))
        # print(x.shape)
        mini_batch_x = [x[:, k*self.test_batch_size:(k+1)*self.test_batch_size]\
                        for k in range(x.shape[1]//self.test_batch_size)]
        if x.shape[1] % self.test_batch_size != 0:
            mini_batch_x.append(x[:, x.shape[1]//self.test_batch_size*self.test_batch_size:])

        y_pred = None
        import time
        start_t = time.time()
        for i in tqdm(range(len(mini_batch_x))):
            if i%10 != 0:
                self.model = self.model.eval()
            else:
                self.model = self.model.train()
            # (BS, I, B) -> (I, BS, B)
            x = np.array(mini_batch_x[i])  # 为了适应LSTM参数：第二个维度是batch_size
            x = torch.from_numpy(x).to(self.device)
            y_,_ = self.model(x)  # jmy
            y = y_.cpu().data.numpy()
            ## modified jmy
            # v1: 实时显示histogram

            predict_acc_each_bin = y_
            y_gt_minibatch = torch.tensor(self.test_y[i:i+1,:],dtype = torch.float32).to(device)
            # print(predict_acc_each_bin.shape,y_gt_minibatch.shape)
            
            if i%10==0:
                test_loss = self.criterion(predict_acc_each_bin.reshape(-1),y_gt_minibatch.reshape(-1))*10
                self.optimizer.zero_grad()
                test_loss.backward()
                self.optimizer.step()
            y_gt_minibatch = y_gt_minibatch.cpu().data.numpy()
            predict_acc_each_bin = predict_acc_each_bin.cpu().data.numpy()

            self.dynamic_plot(predict_acc_each_bin,y_gt_minibatch)
            if i == 0:
                y_pred = y
            else:
                y_pred = np.concatenate([y_pred, y], axis=0)
                
        #阈值判断
        print(f">> The test time is {time.time()-start_t} s")
        y_gt = self.test_y  # (N, O, B)
        conf, Conf = self.compute_conflict(y_pred, y_gt[:,0,:])
        # (time, bin_size) (117, 128)
        # print(conf)
        print(">> Conflict_avoid_rate:", 1 - Conf[3] / Conf[1])
        print(">> Loss:", np.mean(self.test_loss))
        # print(y_pred.shape, y_gt.shape)
        # (117, 128)
        Plot_stft(data=y_pred[:, :].T)
        Plot_stft(data=y_gt[:, 0, :].T)

    def compute_conflict(self, pred, gt): # # (time, bin_size) pred (117, 128) gt
        time, bin_size = pred.shape
        o = np.ones((time, bin_size)) - pred # o代表不占用率
        p_gt = 1/bin_size
        gt_thre = 0.2 # 定义冲突的阈值
        P = 0.5 # 以确定的概率P随机决定发送还是不发送
        conflict1, conflict2, conflict3, conflict4, conflict5 = 0, 0, 0, 0, 0
        Conflict1, Conflict2, Conflict3, Conflict4, Conflict5 = 0, 0, 0, 0, 0
        for i in range(time):
            # p[i][p[i] < 0.6] = -5
            s3 = np.exp(5*o[i]) / np.sum(np.exp(5*o[i]))
            # optimal policy
            s4 = o[i]-np.max(o[i])
            s4 = (s4 == 0).astype(np.float64)
            # 用不占用率来选取bin
            Pc = P*np.sum(np.exp(2*o[i])) / np.sum(np.exp(3*o[i]))
            s5 = np.exp(o[i])

            for j in range(bin_size):
                conflict1 += p_gt * 1 * (gt[i][j] > gt_thre)  # 策略1：随机选bin, 选bin后1概率发送
                conflict2 += p_gt * P * (gt[i][j] > gt_thre) # 策略2：随机选bin, 选bin后P概率发送
                conflict3 += s3[j] * P * (gt[i][j] > gt_thre) # 策略3，按不占用率选bin, 选bin后P概率发送
                conflict4 += s4[j] * P * (gt[i][j] > gt_thre) # 策略4：最优策略，选o最大的bin, 选bin后0P概率发送
                conflict5 += s5[j] * Pc * (gt[i][j] > gt_thre) # 策略5：按不占用率选bin, 选bin后ci概率发送

                # 下面将“冲突”定义成1个概率，即gt的大小
                Conflict1 += p_gt * 1 * gt[i][j]   # 策略1
                Conflict2 += p_gt * P * gt[i][j]  # 策略2
                Conflict3 += s3[j] * P * gt[i][j]  # 策略3
                Conflict4 += s4[j] * P * gt[i][j]  # 策略4
                Conflict5 += s5[j] * Pc * gt[i][j]  # 策略5

        print('conflict1', conflict1 / time, 'conflict2', conflict2 / time, 'conflict3',
              conflict3 / time, 'conflict4', conflict4 / time, 'conflict5', conflict5 / time)
        print('Conflict1', Conflict1 / time, 'Conflict2', Conflict2 / time, 'Conflict3',
              Conflict3 / time, 'Conflict4', Conflict4 / time, 'Conflict5', Conflict5 / time)
        c = np.array([conflict1, conflict2, conflict3, conflict4, conflict5])/time
        C = np.array([Conflict1, Conflict2, Conflict3, Conflict4, Conflict5])/time
        return c, C

    def dynamic_plot(self,pred,gt):  # jmy
        pred = pred.reshape((-1))
        gt = gt.reshape((-1))
        self.test_loss.append( np.sum(np.abs(pred-gt)))
        return
        plt.clf()
        plt.plot(pred,label="prediction")
        plt.plot(gt,label="ground truth")
        plt.legend()
        plt.draw()
        plt.pause(0.001)
