import os
import torch
import yaml
import yamlloader
import numpy as np
import matplotlib.pyplot as plt
import math
import utils.bands_transform as bt
from prettytable import PrettyTable
import matplotlib as mpl

def chunks(lst, n):
    """
    Yield successive n-sized chunks from a list.

    Args:
        lst (list): The input list.
        n (int): The size of each chunk.

    Yields:
        list: A chunk of size n from the input list.

    Examples:
        >>> lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> for chunk in chunks(lst, 3):
        ...     print(chunk)
        [1, 2, 3]
        [4, 5, 6]
        [7, 8, 9]
        [10]
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


class AudioChunks():
    def __init__(self, n, hop, fade=True):
        #number of elements in each chunk
        self.n = n
        #size of hop
        self.hop = hop
        self.diffhop = n - hop

        #whether to apply a fade in and fade out on each chunk or not
        self.fade = fade

    def calculate_num_chunks(self, wavesize):
        num_chunks = 1
        idx = 0
        audio_truncated=False
        
        if self.n == self.diffhop:
            step = self.n
        else:
            step = self.n-self.diffhop

        for i in range(self.n, wavesize-self.n+self.diffhop, step):
            num_chunks += 1
            idx = i

        if idx+2*(self.n-self.diffhop) == wavesize:
            num_chunks += 1
        else:
            audio_truncated=True

        if self.n == self.diffhop:
            if self.n*num_chunks == wavesize:
                audio_truncated=False
            else:
                audio_truncated=True
            
        return(num_chunks, audio_truncated)
    
    def chunks_with_hop(self, lst):
        if isinstance(lst, np.ndarray):
            return self._chunks_with_hop_np(lst)
        elif isinstance(lst, torch.Tensor):
            return self._chunks_with_hop_torch(lst)
        else:
            raise TypeError("Input must be either a numpy array or a torch tensor")

    def _chunks_with_hop_np(self, lst):
        if self.n != self.diffhop:
            L = []
            L.append(lst[0:self.n])
            idx = 0

            if self.n == self.diffhop:
                step = self.n
            else:
                step = self.n - self.diffhop

            for i in range(self.n, len(lst) - self.n + self.diffhop, step):
                L.append(lst[(i - self.diffhop):(i + self.n - self.diffhop)])
                idx = i
            if idx + 2 * (self.n - self.diffhop) == len(lst):
                L.append(lst[len(lst) - self.n:len(lst)])
        else:
            L = []
            step = self.n
            for i in range(0, len(lst), step):
                to_add = lst[i:i + step]
                if len(to_add) == step:
                    L.append(to_add)

        return np.array(L)

    def _chunks_with_hop_torch(self, lst):
        if self.n != self.diffhop:
            L = []
            L.append(lst[0:self.n])
            idx = 0

            if self.n == self.diffhop:
                step = self.n
            else:
                step = self.n - self.diffhop

            for i in range(self.n, len(lst) - self.n + self.diffhop, step):
                L.append(lst[(i - self.diffhop):(i + self.n - self.diffhop)])
                idx = i
            if idx + 2 * (self.n - self.diffhop) == len(lst):
                L.append(lst[len(lst) - self.n:len(lst)])
        else:
            L = []
            step = self.n
            for i in range(0, len(lst), step):
                to_add = lst[i:i + step]
                if len(to_add) == step:
                    L.append(to_add)

        return torch.stack(L)

    def concat_with_hop(self, L):
        if isinstance(L, np.ndarray):
            return self._concat_with_hop_np(L)
        elif isinstance(L, torch.Tensor):
            return self._concat_with_hop_torch(L)
        else:
            raise TypeError("Input must be either a numpy array or a torch tensor")

    def _concat_with_hop_np(self, L):
        lst = np.zeros(shape=L.shape[1] * L.shape[0] - (L.shape[0] - 1) * self.diffhop)
        if self.fade:
            bef = np.linspace(0, 1, self.diffhop)
            aft = np.linspace(1, 0, self.diffhop)
            mid = np.ones(L.shape[1] - self.diffhop)
            pond_g = np.concatenate((bef, mid))
            pond_d = np.concatenate((mid, aft))
        else:
            pond_g = np.ones(L.shape[1])
            pond_d = np.ones(L.shape[1])

        lst[0:L.shape[1]] = pond_d * L[0, :]
        for i in range(1, L.shape[0]):
            if i != L.shape[0] - 1:
                lst[i * L.shape[1] - i * self.diffhop:(i + 1) * L.shape[1] - i * self.diffhop] += pond_g * pond_d * L[i, :]
            else:
                lst[i * L.shape[1] - i * self.diffhop:(i + 1) * L.shape[1] - i * self.diffhop] += pond_g * L[i, :]

        return lst

    def _concat_with_hop_torch(self, L):
        lst = torch.zeros(L.shape[1] * L.shape[0] - (L.shape[0] - 1) * self.diffhop)
        if self.fade:
            bef = torch.linspace(0, 1, self.diffhop)
            aft = torch.linspace(1, 0, self.diffhop)
            mid = torch.ones(L.shape[1] - self.diffhop)
            pond_g = torch.cat((bef, mid))
            pond_d = torch.cat((mid, aft))
        else:
            pond_g = torch.ones(L.shape[1])
            pond_d = torch.ones(L.shape[1])

        lst[0:L.shape[1]] = pond_d * L[0, :]
        for i in range(1, L.shape[0]):
            if i != L.shape[0] - 1:
                lst[i * L.shape[1] - i * self.diffhop:(i + 1) * L.shape[1] - i * self.diffhop] += pond_g * pond_d * L[i, :]
            else:
                lst[i * L.shape[1] - i * self.diffhop:(i + 1) * L.shape[1] - i * self.diffhop] += pond_g * L[i, :]

        return lst

    def concat_spec_with_hop(self, L):
        if isinstance(L, np.ndarray):
            return self._concat_spec_with_hop_np(L)
        elif isinstance(L, torch.Tensor):
            return self._concat_spec_with_hop_torch(L)
        else:
            raise TypeError("Input must be either a numpy array or a torch tensor")

    def _concat_spec_with_hop_np(self, L):
        lst = np.zeros(shape=(L.shape[1], L.shape[2] * L.shape[0] - (L.shape[0] - 1) * self.diffhop))
        if self.fade:
            bef = np.linspace(0, 1, self.diffhop)
            aft = np.linspace(1, 0, self.diffhop)
            mid = np.ones(L.shape[2] - self.diffhop)
            pond_g = np.concatenate((bef, mid))
            pond_d = np.concatenate((mid, aft))

            pond_g = np.tile(pond_g, (L.shape[1], 1))
            pond_d = np.tile(pond_d, (L.shape[1], 1))
        else:
            pond_g = np.ones((L.shape[1], L.shape[2]))
            pond_d = np.ones((L.shape[1], L.shape[2]))

        lst[:, 0:L.shape[2]] = pond_d * L[0, :, :]
        for i in range(1, L.shape[0]):
            if i != L.shape[0] - 1:
                lst[:, i * L.shape[2] - i * self.diffhop:(i + 1) * L.shape[2] - i * self.diffhop] += pond_g * pond_d * L[i, :, :]
            else:
                lst[:, i * L.shape[2] - i * self.diffhop:(i + 1) * L.shape[2] - i * self.diffhop] += pond_g * L[i, :, :]

        return lst

    def _concat_spec_with_hop_torch(self, L):
        lst = torch.zeros((L.shape[1], L.shape[2] * L.shape[0] - (L.shape[0] - 1) * self.diffhop))
        if self.fade:
            bef = torch.linspace(0, 1, self.diffhop)
            aft = torch.linspace(1, 0, self.diffhop)
            mid = torch.ones(L.shape[2] - self.diffhop)
            pond_g = torch.cat((bef, mid))
            pond_d = torch.cat((mid, aft))

            pond_g = pond_g.repeat(L.shape[1], 1)
            pond_d = pond_d.repeat(L.shape[1], 1)
        else:
            pond_g = torch.ones((L.shape[1], L.shape[2]))
            pond_d = torch.ones((L.shape[1], L.shape[2]))

        lst[:, 0:L.shape[2]] = pond_d * L[0, :, :]
        for i in range(1, L.shape[0]):
            if i != L.shape[0] - 1:
                lst[:, i * L.shape[2] - i * self.diffhop:(i + 1) * L.shape[2] - i * self.diffhop] += pond_g * pond_d * L[i, :, :]
            else:
                lst[:, i * L.shape[2] - i * self.diffhop:(i + 1) * L.shape[2] - i * self.diffhop] += pond_g * L[i, :, :]

        return lst






















class AudioChunksOld():
    def __init__(self, n, hop, fade=True):
        #number of elements in each chunk
        self.n = n
        #size of hop
        self.hop = hop
        self.diffhop = n - hop

        #whether to apply a fade in and fade out on each chunk or not
        self.fade=fade

    def chunks_with_hop(self, lst):

        if self.n != self.diffhop:
            L = []
            L.append(lst[0:self.n])
            idx = 0

            if self.n == self.diffhop:
                step = self.n
            else:
                step = self.n-self.diffhop

            for i in range(self.n, len(lst)-self.n+self.diffhop, step):
                # print('FFFFFFFFFFF')
                # print(lst[(i-self.hop):(i + self.n - self.hop)])
                L.append(lst[(i-self.diffhop):(i + self.n - self.diffhop)])
                idx = i
            if idx+2*(self.n-self.diffhop) == len(lst):
                L.append(lst[len(lst)-self.n:len(lst)])
        else:
            L = []
            step = self.n
            for i in range(0, len(lst), step):
                to_add = lst[i:i + step]
                if len(to_add) == step:
                    L.append(to_add)

        return(np.array(L))

    def concat_with_hop(self, L):
        lst = np.zeros(shape=L.shape[1]*L.shape[0] - (L.shape[0]-1)*self.diffhop)
        if self.fade:
            bef = np.linspace(0, 1, self.diffhop)
            aft = np.linspace(1, 0, self.diffhop)
            mid = np.ones(L.shape[1]-self.diffhop)
            pond_g = np.concatenate((bef, mid))
            pond_d = np.concatenate((mid, aft))
        else:
            pond_g = np.ones(L.shape[1])
            pond_d = np.ones(L.shape[1])

        lst[0:L.shape[1]] = pond_d*L[0, :]
        for i in range (1, L.shape[0]):
            if i != L.shape[0]-1:
                lst[i*L.shape[1]-i*self.diffhop:(i+1)*L.shape[1]-i*self.diffhop] += pond_g*pond_d*L[i, :]
            else:
                lst[i*L.shape[1]-i*self.diffhop:(i+1)*L.shape[1]-i*self.diffhop] += pond_g*L[i, :]
        
        return(lst)

    def concat_spec_with_hop(self, L):
        #dim of L: 13, 513, 128
        lst = np.zeros(shape=(L.shape[1], L.shape[2]*L.shape[0] - (L.shape[0]-1)*self.diffhop))
        if self.fade:
            bef = np.linspace(0, 1, self.diffhop)
            aft = np.linspace(1, 0, self.diffhop)
            mid = np.ones(L.shape[2]-self.diffhop)
            pond_g = np.concatenate((bef, mid))
            pond_d = np.concatenate((mid, aft))

            pond_g = np.tile(pond_g, (L.shape[1], 1))
            pond_d = np.tile(pond_d, (L.shape[1], 1))
        else:
            pond_g = np.ones((L.shape[1], L.shape[2]))
            pond_d = np.ones((L.shape[1], L.shape[2]))
        
        lst[:, 0:L.shape[2]] = pond_d*L[0, :, :]
        for i in range (1, L.shape[0]):
            if i != L.shape[0]-1:
                lst[:, i*L.shape[2]-i*self.diffhop:(i+1)*L.shape[2]-i*self.diffhop] += pond_g*pond_d*L[i, :, :]
            else:
                lst[:, i*L.shape[2]-i*self.diffhop:(i+1)*L.shape[2]-i*self.diffhop] += pond_g*L[i, :, :]
        
        return(lst)
    
    # def concat_spec_with_hop(self, L):
    #     #dim of L: 13, 513, 128
    #     lst = np.zeros(shape=(L.shape[1], L.shape[2]*L.shape[0] - (L.shape[0]-1)*self.diffhop))
    #     if self.fade:
    #         bef = np.linspace(0, 1, self.diffhop)
    #         aft = np.linspace(1, 0, self.diffhop)
    #         mid = np.ones(L.shape[2]-self.diffhop)
    #         pond_g = np.concatenate((bef, mid))
    #         pond_d = np.concatenate((mid, aft))

    #         pond_g = np.tile(pond_g, (L.shape[1], 1))
    #         pond_d = np.tile(pond_d, (L.shape[1], 1))
    #     else:
    #         pond_g = np.ones((L.shape[1], L.shape[2]))
    #         pond_d = np.ones((L.shape[1], L.shape[2]))
        
    #     lst[:, 0:L.shape[2]] = pond_d*L[0, :, :]
    #     for i in range (1, L.shape[0]):
    #         if i != L.shape[0]-1:
    #             lst[:, i*L.shape[2]-i*self.diffhop:(i+1)*L.shape[2]-i*self.diffhop] += pond_g*pond_d*L[i, :, :]
    #         else:
    #             lst[:, i*L.shape[2]-i*self.diffhop:(i+1)*L.shape[2]-i*self.diffhop] += pond_g*L[i, :, :]
        
    #     return(lst)
    
    def calculate_num_chunks(self, wavesize):
        num_chunks = 1
        idx = 0
        audio_truncated=False
        
        if self.n == self.diffhop:
            step = self.n
        else:
            step = self.n-self.diffhop

        for i in range(self.n, wavesize-self.n+self.diffhop, step):
            num_chunks += 1
            idx = i

        if idx+2*(self.n-self.diffhop) == wavesize:
            num_chunks += 1
        else:
            audio_truncated=True

        if self.n == self.diffhop:
            if self.n*num_chunks == wavesize:
                audio_truncated=False
            else:
                audio_truncated=True
            
        return(num_chunks, audio_truncated)

class SettingsLoader(yaml.SafeLoader):
    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        super(SettingsLoader, self).__init__(stream)
    
    def include(self, node):
        filename = os.path.join(self._root, self.construct_scalar(node))
        with open(filename, 'r') as f:
            #return yaml.load(f, YAMLLoader)
            return yaml.load(f, yamlloader)
SettingsLoader.add_constructor('!include', SettingsLoader.include)

def load_settings(file_path):
    with file_path.open('r') as f:
        return yaml.load(f, Loader=SettingsLoader)
    
def tukey_window(M, alpha=0.2):
    """Return a Tukey window, also known as a tapered cosine window, and an 
    energy correction value to make sure to preserve energy.
    Window and energy correction calculated according to:
    https://github.com/nicolas-f/noisesensor/blob/master/core/src/acoustic_indicators.c#L150

    Parameters
    ----------
    M : int
        Number of points in the output window. 
    alpha : float, optional
        Shape parameter of the Tukey window, representing the fraction of the
        window inside the cosine tapered region.
        If zero, the Tukey window is equivalent to a rectangular window.
        If one, the Tukey window is equivalent to a Hann window.

    Returns
    -------
    window : ndarray
        The window, with the maximum value normalized to 1.
    energy_correction : float
        The energy_correction used to compensate the loss of energy due to
        the windowing
    """
    #nicolas' calculation
    index_begin_flat = int((alpha / 2) * M)
    index_end_flat = int(M - index_begin_flat)
    energy_correction = 0
    window = np.zeros(M)
    
    for i in range(index_begin_flat):
        window_value = (0.5 * (1 + math.cos(2 * math.pi / alpha * ((i / M) - alpha / 2))))
        energy_correction += window_value * window_value
        window[i]=window_value
    
    energy_correction += (index_end_flat - index_begin_flat) #window*window=1
    for i in range(index_begin_flat, index_end_flat):
        window[i] = 1
    
    for i in range(index_end_flat, M):
        window_value = (0.5 * (1 + math.cos(2 * math.pi / alpha * ((i / M) - 1 + alpha / 2))))
        energy_correction += window_value * window_value
        window[i] = window_value
    
    energy_correction = 1 / math.sqrt(energy_correction / M)
    
    return(window, energy_correction)

def get_transforms(sr=32000, flen=4096, hlen=4000, classifier='YamNet', device=torch.device("cpu"), tho_freq=True, tho_time=True, mel_template=None):
    if mel_template is None:
        tho_tr = bt.ThirdOctaveTransform(sr=sr, flen=flen, hlen=hlen)
        if classifier == 'PANN':
            mels_tr = bt.PANNMelsTransform(flen_tho=tho_tr.flen, hlen_tho=tho_tr.hlen, device=device)
        if classifier == 'YamNet':
            mels_tr = bt.YamNetMelsTransform(flen_tho=tho_tr.flen, hlen_tho=tho_tr.hlen, device=device)
        if classifier == 'default':
            mels_tr = bt.DefaultMelsTransform(sr=tho_tr.sr, flen=tho_tr.flen, hlen=tho_tr.hlen)
    else:
        tho_tr = bt.NewThirdOctaveTransform(32000, 1024, 320, 64, mel_template=mel_template, tho_freq=tho_freq, tho_time=tho_time)
        if classifier == 'PANN':
            mels_tr = bt.PANNMelsTransform(flen_tho=4096, device=device)
        if classifier == 'YamNet':
            mels_tr = bt.YamNetMelsTransform(flen_tho=4096, device=device)
        if classifier == 'default':
            mels_tr = bt.DefaultMelsTransform(sr=tho_tr.sr, flen=4096, hlen=4000)
    return(tho_tr, mels_tr)   

#count the number of parameters of a model
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def plot_spectro(x_m, fs, title='title', vmin=None, vmax=None, diff=False, name='default', ylabel='Mel bin', save=False, save_path='figures_spectrograms/', show_info=True, plot=True):
    if vmin==None:
        vmin = torch.min(x_m)
    if vmax==None:
        vmax = torch.max(x_m)
    exthmin = 1
    exthmax = len(x_m)
    extlmin = 0
    #extlmax = 1
    extlmax = len(x_m[0])/100

    if show_info:
        plt.figure(figsize=(8, 5))
    else:
        # # Remove the frame around the plot
        # plt.gca().spines['top'].set_visible(False)
        # plt.gca().spines['right'].set_visible(False)
        # plt.gca().spines['bottom'].set_visible(False)
        # plt.gca().spines['left'].set_visible(False)

        # # Make the image take the full screen
        # plt.figure(figsize=(1, 1))
        plt.figure(figsize=(8, 5))

    if diff:
        plt.imshow(x_m, extent=[extlmin,extlmax,exthmin,exthmax], cmap='seismic',
                vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        if show_info:
            cbar = plt.colorbar()
            cbar.ax.set_ylabel('Power differences (dB)', rotation=90, labelpad=15)
    else:
        plt.imshow(x_m, extent=[extlmin,extlmax,exthmin,exthmax], cmap='inferno',
                vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        
        if show_info:
            cbar = plt.colorbar()
            cbar.ax.set_ylabel('Power (dB)', rotation=90, labelpad=15)

    if show_info:
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel(ylabel)

    if save:
        plt.savefig(save_path+name)
    if plot:
        plt.show()

def plot_raw_spectro(x_m, vmin=None, vmax=None, diff=False, name='default', ylabel='Mel bin', save=True, save_path='figures/', plot=False):
    if vmin is None:
        vmin = np.min(x_m)
    if vmax is None:
        vmax = np.max(x_m)
    exthmin = 1
    exthmax = len(x_m)
    extlmin = 0
    extlmax = len(x_m[0]) / 100

    plt.figure(figsize=(8, 8))

    if diff:
        plt.imshow(x_m, extent=[extlmin, extlmax, exthmin, exthmax], cmap='seismic',
                   vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
    else:
        plt.imshow(x_m, extent=[extlmin, extlmax, exthmin, exthmax], cmap='inferno',
                   vmin=vmin, vmax=vmax, origin='lower', aspect='auto')

    # Remove ticks, labels, and colorbar
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    if save:
        plt.savefig(save_path + name, bbox_inches='tight', pad_inches=0)
    if plot:
        plt.show()

def plot_multi_spectro(x_m, fs, title='title', vmin=None, vmax=None, diff=False, name='default', ylabel='Mel bin', save=False, extlmax=None, show_colorbar=True):
    if isinstance(x_m[0], torch.Tensor):
        if vmin is None:
            vmin = torch.min(x_m[0])
        if vmax is None:
            vmax = torch.max(x_m[0])
    elif isinstance(x_m[0], np.ndarray):
        if vmin is None:
            vmin = np.min(x_m[0])
        if vmax is None:
            vmax = np.max(x_m[0])
    elif isinstance(x_m[0], list):
        if vmin is None:
            vmin = min(x_m[0])
        if vmax is None:
            vmax = max(x_m[0])

    exthmin = 1
    exthmax = len(x_m)
    extlmin = 0
    if extlmax==None:
        extlmax = len(x_m[0])

    font_size = 40
    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = font_size
    #mpl.use("pgf")
    # mpl.rcParams.update({
    #     "pgf.texsystem": "pdflatex",
    #     'font.family': 'Times New Roman',
    #     'text.usetex': True,
    #     'pgf.rcfonts': False,
    # })

    #fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharey=True, gridspec_kw={'width_ratios': [1, 1, 1, 1]})
    fig, axs = plt.subplots(ncols=len(x_m), figsize=(len(x_m)*10, 10))
    #fig.subplots_adjust(wspace=1)
    xlabel = 'Time (s)'
    for i, ax in enumerate(axs):
        if type(ylabel) is list:
            exthmin = 1
            exthmax = len(x_m[i])
            ylabel_ = ylabel[i] 
        else:
            if i == 0:
                ylabel_ = ylabel
            else:
                ylabel_ = ''
        if diff:
            im = ax.imshow(x_m[i], extent=[extlmin,extlmax,exthmin,exthmax], cmap='seismic',
                    vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        else:
            im = ax.imshow(x_m[i], extent=[extlmin,extlmax,exthmin,exthmax], cmap='inferno',
                    vmin=vmin, vmax=vmax, origin='lower', aspect='auto')

        ax.set_title(title[i])
        #ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabel_)
        ax.set_xlabel(xlabel)
    # fig.text(0.5, 0.1, 'Time (s)', ha='center', va='center')
    
    #cbar_ax = fig.add_axes([0.06, 0.15, 0.01, 0.7])
    if show_colorbar:
        cbar_ax = fig.add_axes([0.97, 0.15, 0.01, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax, label='Power (dB)')
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.yaxis.set_ticks_position('left')

    # if type(ylabel) is list:
    #     for ax, lab in zip(axs, ylabel):
    #         ax.set_ylabel(lab)
    # else:
    #     axs[0].set_ylabel(ylabel)

    #fig.tight_layout()
    #fig.tight_layout(rect=[0.1, 0.05, 1, 1], pad=2)
    fig.tight_layout(rect=[0, 0.05, 0.92, 1], pad=2)
    #fig.savefig('fig_spectro' + name + '.pdf', bbox_inches='tight', dpi=fig.dpi)
    if save:
        plt.savefig('figures/' + name + '.pdf', dpi=fig.dpi, bbox_inches='tight')
    else:
        plt.show()

def plot_multi_spectro_with_thirdo(x_m, fs, title='title', vmin=None, vmax=None, diff=False, name='default', ylabel='Mel bin', save=False):
    if vmin is None:
        vmin = torch.min(x_m)
    if vmax is None:
        vmax = torch.max(x_m)
    
    exthmin = 1
    exthmax = len(x_m[0])
    extlmin = 0
    extlmax = 1

    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 20

    fig, axs = plt.subplots(ncols=len(x_m), sharey=True, figsize=(len(x_m) * 9, 5), gridspec_kw={'width_ratios': [2, 2, 2, 2]})
    # plt.subplots_adjust(left=0.1)

    for i, ax in enumerate(axs):
        if i == 0:
            continue
        if i == 1:
            ylabel_ = ylabel
            xlabel_ = "Time (s)"
        else:
            ylabel_ = ''
            xlabel_ = ""
        if diff:
            im = ax.imshow(x_m[i], extent=[extlmin, extlmax, exthmin, exthmax], cmap='seismic',
                    vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        else:
            im = ax.imshow(x_m[i], extent=[extlmin, extlmax, exthmin, exthmax], cmap='inferno',
                    vmin=vmin, vmax=vmax, origin='lower', aspect='auto')

        ax.set_title(title[i])
        ax.set_ylabel(ylabel_)
        ax.set_xlabel(xlabel_)
        ax.set_position([ax.get_position().xmin, ax.get_position().ymin, ax.get_position().width, 0.7])

    # Adding the colorbar for the last subplot
    cbar_ax = fig.add_axes([0.97, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, label='Power (dB)')
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.yaxis.set_ticks_position('left')

    # axs[0].set_ylabel('third-octave bins')

    # Adding the first element with its own ticks on the x and y axes
    first_element_ax = axs[0]
    vmin_0 = np.min(x_m[0])
    vmax_0 = np.max(x_m[0])
    im = first_element_ax.imshow(x_m[0], extent=[extlmin, extlmax, exthmin, exthmax], cmap='inferno',
            vmin=vmin_0, vmax=vmax_0, origin='lower', aspect='auto')
    first_element_ax.set_title(title[0])
    first_element_ax.set_ylabel('Third-octave bin')
    first_element_ax.set_xlabel('Time (s)')
    # first_element_ax.set_position([0.1, 0.15, 0.01, 0.7])
    first_element_ax.set_position([0.04, axs[1].get_position().ymin, axs[1].get_position().width, axs[1].get_position().height])

    # Adding a separate colorbar for the first element on the left
    first_cbar_ax = fig.add_axes([0.27, 0.15, 0.01, 0.7])
    first_cbar = fig.colorbar(im, cax=first_cbar_ax, label='Power (dB)')
    first_cbar.ax.yaxis.set_label_position('left')
    first_cbar.ax.yaxis.set_ticks_position('left')

    # fig.tight_layout(rect=[0.1, 0.05, 0.9, 1], pad=2)

    if save:
        plt.savefig('fig_spectro' + name + '.pdf', dpi=fig.dpi, bbox_inches='tight')
    plt.show()


def sort_labels_by_score(scores, labels, top=-1):
    # create a list of tuples where each tuple contains a score and its corresponding label
    score_label_tuples = list(zip(scores, labels))
    # sort the tuples based on the score in descending order
    sorted_tuples = sorted(score_label_tuples, reverse=True)
    # extract the sorted labels from the sorted tuples
    sorted_labels = [t[1] for t in sorted_tuples]
    sorted_scores = [t[0] for t in sorted_tuples]
    
    # create a list of 1s and 0s indicating if the score is in the top 10 or not
    top_scores = sorted_scores[:top]
    top_labels = sorted_labels[:top]

    if top >= 1:
        in_top = [1 if label in top_labels else 0 for label in labels]
    else:
        in_top = None
    
    return sorted_scores, sorted_labels, in_top

# #MT: added
# class ChunkManager():
#     def __init__(self, dataset_name, model_name, model_batch_path, batch_type_name, batch_lim=1000):
#         self.dataset_name = dataset_name
#         self.model_name = model_name
#         self.model_batch_path = model_batch_path
#         self.batch_type_name = batch_type_name
#         self.current_batch_id = 0
#         self.batch_lim = batch_lim
#         self.total_folder_name = self.batch_type_name + '_' + self.model_name + '_' + self.dataset_name
#         self.total_path = self.model_batch_path / self.total_folder_name
#         if not os.path.exists(self.total_path):
#             os.makedirs(self.total_path)
#         else:
#             print(f'WARNING: everything will be deleted in path: {self.total_path}')
#             self._delete_everything_in_folder(self.total_path)
        
#     def save_chunk(self, batch, forced=False):
#         if len(batch) == 0:
#             return(batch)
#         if len(batch) >= self.batch_lim or forced == True:
#             file_path = self.total_path / (self.total_folder_name + '_' + str(self.current_batch_id) + '.npy')
#             np.save(file_path, batch)
#             print(f'save made in: {file_path}')
#             self.current_batch_id+=1
#             return([])
#         else:
#             return(batch)
        
#     def open_chunks(self):
#         stacked_batch = np.array([])
#         for root, dirs, files in os.walk(self.total_path):
            
#             #sort files
#             files_splitted = [re.split(r'[_.]', file) for file in files]
#             file_indices = [int(file[-2]) for file in files_splitted]
#             file_indices_sorted = file_indices.copy()
#             file_indices_sorted.sort()
#             file_new_indices = [file_indices.index(ind) for ind in file_indices_sorted]
#             files_sorted = [files[i] for i in file_new_indices]
            

#             for file in files_sorted:
#                 cur_batch = np.load(self.total_path / file, allow_pickle=True)
#                 stacked_batch = np.concatenate((stacked_batch, cur_batch))

#         return(stacked_batch)
            
#     def _delete_everything_in_folder(self, folder):
#         for filename in os.listdir(folder):
#             file_path = os.path.join(folder, filename)
#             try:
#                 if os.path.isfile(file_path) or os.path.islink(file_path):
#                     os.unlink(file_path)
#                 elif os.path.isdir(file_path):
#                     send2trash(file_path)
#                     #shutil.rmtree(file_path)
#             except Exception as e:
#                 print('Failed to delete %s. Reason: %s' % (file_path, e))



#UTILS FROM PANN

def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)
        
        
def get_filename(path):
    path = os.path.realpath(path)
    na_ext = path.split('/')[-1]
    na = os.path.splitext(na_ext)[0]
    return na


def get_sub_filepaths(folder):
    paths = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            paths.append(path)
    return paths
    
    
def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1
        
    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging


def read_metadata(csv_path, classes_num, id_to_ix):
    """Read metadata of AudioSet from a csv file.

    Args:
      csv_path: str

    Returns:
      meta_dict: {'audio_name': (audios_num,), 'target': (audios_num, classes_num)}
    """

    with open(csv_path, 'r') as fr:
        lines = fr.readlines()
        lines = lines[3:]   # Remove heads

    audios_num = len(lines)
    targets = np.zeros((audios_num, classes_num), dtype=np.bool)
    audio_names = []
 
    for n, line in enumerate(lines):
        items = line.split(', ')
        """items: ['--4gqARaEJE', '0.000', '10.000', '"/m/068hy,/m/07q6cd_,/m/0bt9lr,/m/0jbk"\n']"""

        audio_name = 'Y{}.wav'.format(items[0])   # Audios are started with an extra 'Y' when downloading
        label_ids = items[3].split('"')[1].split(',')

        audio_names.append(audio_name)

        # Target
        for id in label_ids:
            ix = id_to_ix[id]
            targets[n, ix] = 1
    
    meta_dict = {'audio_name': np.array(audio_names), 'target': targets}
    return meta_dict


def float32_to_int16(x):
    assert np.max(np.abs(x)) <= 1.2
    x = np.clip(x, -1, 1)
    return (x * 32767.).astype(np.int16)

def int16_to_float32(x):
    return (x / 32767.).astype(np.float32)
    

def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x))), axis=0)
    else:
        return x[0 : audio_length]


def d_prime(auc):
    d_prime = stats.norm().ppf(auc) * np.sqrt(2.0)
    return d_prime


class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        """Mixup coefficient generator.
        """
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        """Get mixup random coefficients.
        Args:
          batch_size: int
        Returns:
          mixup_lambdas: (batch_size,)
        """
        mixup_lambdas = []
        for n in range(0, batch_size, 2):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1. - lam)

        return np.array(mixup_lambdas)


class StatisticsContainer(object):
    def __init__(self, statistics_path):
        """Contain statistics of different training iterations.
        """
        self.statistics_path = statistics_path

        self.backup_statistics_path = '{}_{}.pkl'.format(
            os.path.splitext(self.statistics_path)[0], 
            datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        self.statistics_dict = {'bal': [], 'test': []}

    def append(self, iteration, statistics, data_type):
        statistics['iteration'] = iteration
        self.statistics_dict[data_type].append(statistics)
        
    def dump(self):
        pickle.dump(self.statistics_dict, open(self.statistics_path, 'wb'))
        pickle.dump(self.statistics_dict, open(self.backup_statistics_path, 'wb'))
        logging.info('    Dump statistics to {}'.format(self.statistics_path))
        logging.info('    Dump statistics to {}'.format(self.backup_statistics_path))
        
    def load_state_dict(self, resume_iteration):
        self.statistics_dict = pickle.load(open(self.statistics_path, 'rb'))

        resume_statistics_dict = {'bal': [], 'test': []}
        
        for key in self.statistics_dict.keys():
            for statistics in self.statistics_dict[key]:
                if statistics['iteration'] <= resume_iteration:
                    resume_statistics_dict[key].append(statistics)
                
        self.statistics_dict = resume_statistics_dict