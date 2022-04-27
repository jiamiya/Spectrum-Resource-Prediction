import numpy as np
from scipy.signal import stft, medfilt
from scipy.io import loadmat
from util.plot_tools import Plot_stft, Plot_signal

# C是数据条数(如每个mat由10条数据)
# B是频率轴上的长度(即bin_size)
# T是时间轴上的长度
# 做数据预处理的话，只要修改DataLoader.preprocess，同时，保证self.data_pro的shape=（C, B, T）形式即可


class DataLoader:
    # 读取数据
    def __init__(self,
                 data_dir,
                 bin_size,
                 sample_rate,data_dir2=None,using_group10_data = False):
        print("-----<Data Loader>------")
        self.using_group10_data = using_group10_data
        if not using_group10_data:
            self.data_raw = loadmat(data_dir)['data'][:, :]  # (1, T)100000000
            self.bin_size = bin_size  # bin的个数
            self.sample_rate = sample_rate  # 采样频率
            # 做stft后的数据 array (C, B, T)
            print(">> Now we are applying STFT to raw data, which will take a while if the data is large...")
            _, _, self.data_stft = stft(self.data_raw,
                                        sample_rate,
                                        nperseg=bin_size,
                                        return_onesided=False)
            self.data_stft = self.data_stft.transpose((0,2,1)).reshape((1,self.data_stft.shape[-1],-1,16))
        else:  # 取recovered数据前70%作为训练，取original数据后30%作为测试，将两部分拼成data_raw #jmy
            self.data_raw = np.load(data_dir)
            self.data_raw = self.data_raw[:int(self.data_raw.shape[0]*0.49),:]
            data_origin = np.load(data_dir2)
            data_origin = (data_origin[int(data_origin.shape[0]*0.49):int(data_origin.shape[0]*(0.7)),:]-data_origin.mean())/data_origin.std()
            self.data_raw = np.concatenate((self.data_raw,data_origin))
            self.bin_size = bin_size  # bin的个数
            self.sample_rate = sample_rate  # 采样频率
            self.data_stft = self.data_raw.reshape((1,self.data_raw.shape[0],-1,16))
            # print(self.data_stft.shape)
        self.bin_size /= 16
        self.data_stft = np.mean(self.data_stft,axis=3)
        self.data_stft = self.data_stft.transpose((0,2,1))

        print("data_raw [ndarray]:", self.data_raw.shape)
        print("bin_size:", bin_size)
        print("sample_rate:", sample_rate)
        print("data_stft [ndarray]:", self.data_stft.shape)
        self.data_pro = None  # 预处理后的数据
        

    def preprocess(self):
        # 数据预处理， shape(self.data_stft) = (C, B, T)
        print("-----<Data Process>------")
        stft_abs = np.abs(self.data_stft)  # 对stft后的数据做abs来去掉虚部, (C, B, T)

        stft_ft = []
        for i in range(stft_abs.shape[0]):
            filterd = np.array([medfilt(row, 1) for row in stft_abs[i]]) #中值滤波
            stft_ft.append(filterd)
        stft_ft = np.array(stft_ft)  # 经过滤波后的数据 (C, B, T)
        if not self.using_group10_data:
            stft_db = 20*np.log10(stft_ft)-10*np.log10(self.bin_size)+10*np.log10(self.sample_rate)-30  # 转换到db单位空间
            stft_pro = (stft_db>-65 ).astype(np.float32) #(stft_zo > 0.60).astype(np.float32)  # 阈值处理 #modified jmy
        else:
            stft_pro = (stft_ft - stft_ft.min()) / (stft_ft.max() - stft_ft.min())
            stft_pro = (stft_pro>1.0*stft_pro.mean()).astype(np.float32)

        self.data_pro = stft_pro  # （C, B, T）
        print("data_pro: ", self.data_pro.shape)

    def merge_data(self,data):
        test_num = data.shape[2]
        self.data_pro = np.concatenate((self.data_pro,data),axis=2)
        return test_num/self.data_pro.shape[2]


