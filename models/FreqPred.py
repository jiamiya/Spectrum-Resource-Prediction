import torch
from torch import nn
import numpy as np
from tqdm import tqdm,trange
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

        output = torch.mean(output,dim=0)
        
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
        return x.transpose((1, 0, 2, 3)), y  #.transpose((1, 0, 2, 3))  # (C, T, I/O, B)

    def build(self, batch_size, lr, num_layer, weight_name='', load_dir=''):
        # 网络初始化
        self.base_dir = Path(os.path.join('logs', 'model'))
        self.base_dir.mkdir(parents=True, exist_ok=True)
        if weight_name=='':
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
                          input_len=self.in_len,
                          output_len=self.out_len
                          ).to(self.device)
        if load_dir!='':
            weight = torch.load(os.path.join(self.base_dir, load_dir))
            self.model.load_state_dict(weight)
            print(">> Model_weight is loaded.")
        # self.criterion = nn.MSELoss().to(self.device)
        self.criterion = nn.L1Loss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

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
        import time
        start_time = time.time()
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
        print(time.time()-start_time)
        y_gt = self.test_y  # (N, O, B)
        acc = self.compute_acc(y_pred, y_gt)
        conflict_rate1, conflict_rate2 = self.compute_conflict(y_pred, y_gt[:,0,:])
        # (time, bin_size) (117, 128)
        print(">> Acc:", acc)
        print(">> Conflict_avoid_rate:", 1-conflict_rate2/ conflict_rate1)
        print("loss:",np.mean(self.test_loss))
        print(y_pred.shape, y_gt.shape)
        # yyy = y_pred.transpose(1, 0, 2).reshape((-1, 256))
        # (117, 128)
        Plot_stft(data=y_pred[:, :].T)
        Plot_stft(data=y_gt[:, 0, :].T)

    def compute_acc(self, pred, gt):
        N, O, B = gt.shape  # (N, O, B)
        return np.sum((pred == gt))/(N * O * B)

    # def compute_conflict(self, pred, gt): # # (time, bin_size) pred (117, 128) gt
    #     time, bin_size = pred.shape
    #     p = np.ones((time, bin_size)) - pred # p代表不占用率
    #     p_gt = 1/bin_size
    #     ave_conflict1, ave_conflict2, ave_conflict3, ave_conflict4 = 0, 0, 0, 0
    #     gt_thre = 0.3
    #     for i in range(time):
    #         # p[i][p[i] < 0.6] = -5
    #         # s = np.exp(20*p[i]) / np.sum(np.exp(20*p[i]))
    #         # optimal policy
    #         s = p[i]-np.max(p[i])
    #         s = (s==0).astype(np.float64)
    #         # 用不占用率来选取bin
    #         conflict1, conflict2, conflict3, conflict4 = 0, 0, 0, 0
    #         for j in range(bin_size):
    #             # 预期conflict1>conflict2, conflict3>conflict4
    #             # 实际上如果不用softmax可能会conflict1<conflict2, 用了softmax后conflict1>conflict2但差距不大
    #             # 感觉问题出在(gt[i][j] > gt_thre)，即“冲突”的定义上
    #             # 选哪个bin * 这个bin里面发不发* 这个bin里发送冲不冲突
    #             conflict1 += p_gt * 0.5 * (gt[i][j] > gt_thre) # 随机选bin, 选bin后0.5概率发送
    #             conflict2 += s[j] * 0.5 * (gt[i][j] > gt_thre) # 按不占用率选bin, 选bin后0.5概率发送
    #             # conflict2 += ((1-pred[i][j])/total) * (1-pred[i][j])*(gt[i][j] > gt_thre)
    #             # 后期考虑将 conflict2 的0.5改成与 (1-pred[i][j]) 正相关，但要做好归一化
    #             # 如果这样改了——按不占用率选bin, 选bin后某个与不占用率相关的概率发送

    #             conflict3 += p_gt * gt[i][j] # 随机选bin, “冲突”定义成1个概率，即gt的大小
    #             conflict4 += s[j] * gt[i][j] # 按不占用率选bin, “冲突”定义成1个概率，即gt的大小
    #         # print(conflict1, conflict2, conflict3, conflict4)
    #         ave_conflict1 += conflict1
    #         ave_conflict2 += conflict2
    #         ave_conflict3 += conflict3
    #         ave_conflict4 += conflict4

    #     print('conflict1', ave_conflict1 / time, 'conflict2', ave_conflict2 / time)
    #     print('conflict3', ave_conflict3/time, 'conflict4', ave_conflict4/time)
    #     return ave_conflict3/time, ave_conflict4/time

    def compute_conflict(self, pred, gt): # # (time, bin_size) pred (117, 128) gt
        time, bin_size = pred.shape
        o = np.ones((time, bin_size)) - pred # o代表不占用率
        p_gt = 1/bin_size
        gt_thre = 0.3 # 定义冲突的阈值
        P = 0.5 # 以确定的概率P随机决定发送还是不发送
        conflict1, conflict2, conflict3, conflict4, conflict5 = 0, 0, 0, 0, 0
        Conflict1, Conflict2, Conflict3, Conflict4, Conflict5 = 0, 0, 0, 0, 0
        for i in range(time):
            # p[i][p[i] < 0.6] = -5
            s3 = np.exp(5*o[i]) / np.sum(np.exp(5*o[i]))
            # optimal policy
            s4 = o[i]-np.max(o[i])
            s4 = (s4==0).astype(np.float64)
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

    def dynamic_plot(self,pred,gt):# jmy
        # print("#",pred.shape,gt.shape)
        pred = pred.reshape((-1))
        gt = gt.reshape((-1))
        self.test_loss.append( np.sum(np.abs(pred-gt)))
        return
        # return
        # pred = np.sum(pred,axis=1)
        # pred = np.sum(pred,axis=0)
        # now pred shape is (256,)
        # gt = np.sum(gt,axis=1)
        # gt = np.sum(gt,axis=0)
        plt.clf()
        plt.plot(pred,label="prediction")
        plt.plot(gt,label="ground truth")
        # plt.ylim((0,1))
        plt.legend()
        plt.draw()
        plt.pause(0.001)
