import matplotlib.pyplot as plt
import numpy as np


def Plot_stft(data,
              sample_rate=1,
              title='STFT',
              save_name=''):
    plt.pcolormesh(data) # data: (bin_size, time)
    plt.colorbar()
    plt.title(title)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Sample Point (Sample Rate = '+str(sample_rate)+' )')
    plt.tight_layout()
    if save_name != '':
        plt.savefig(save_name)
    plt.show()
    return


def Plot_signal(data,
                sample_rate=1,
                title='One Bin Magnitude',
                xlabel='Time',
                ylabel='Magnitude',
                save_name=''):
    t = np.arange(data.shape[0])/sample_rate
    plt.plot(t, data)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.tight_layout()
    if save_name != '':
        plt.savefig(save_name)
    plt.show()
    return
