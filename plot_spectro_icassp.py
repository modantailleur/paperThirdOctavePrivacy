
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 15:02:54 2022

@author: user
"""
import numpy as np
import librosa
import numpy as np
import librosa
import torch.utils.data
import torch
from transcoders import ThirdOctaveToMelTranscoderPinv, ThirdOctaveToMelTranscoder
from utils.util import sort_labels_by_score
from utils.util import plot_multi_spectro, plot_raw_spectro
import utils.bands_transform as bt

def add_white_noise(array, noise_level=0.01):
    """
    Adds white noise to a 2D NumPy array with noise values between 0 and 1.

    Parameters:
    array (np.ndarray): The input 2D array to which noise will be added.
    noise_level (float): The maximum value of the noise to be added. Default is 0.01.

    Returns:
    np.ndarray: The array with added white noise.
    """
    # Generate white noise between 0 and 1
    noise = np.random.rand(*array.shape)
    
    # Add the noise to the original array
    noisy_array = array*(1-noise_level) + noise*noise_level
    
    return noisy_array

if __name__ == '__main__':
    plot_audio_path = './audio/for_plot/'
    x_oracle_24k = librosa.load(plot_audio_path + 'evalset=ljspeech+method=oracle+step=vocode+tho_type=fast___gomin___LJ002-0068.wav', sr=24000)[0]
    x_oracle_32k = librosa.load(plot_audio_path + 'evalset=ljspeech+method=oracle+step=vocode+tho_type=fast___gomin___LJ002-0068.wav', sr=32000)[0]
    x_diff_24k = librosa.load(plot_audio_path + 'dataset=ljspeech+diff_steps=1000+epoch=40+evalset=ljspeech+learning_rate=-4+method=diffusion+schedule=DDPM+step=vocode+tho_type=fast___gomin___LJ002-0068.wav', sr=24000)[0]
    x_diff_32k = librosa.load(plot_audio_path + 'dataset=ljspeech+diff_steps=1000+epoch=40+evalset=ljspeech+learning_rate=-4+method=diffusion+schedule=DDPM+step=vocode+tho_type=fast___gomin___LJ002-0068.wav', sr=32000)[0]
    x_pinv_24k = librosa.load(plot_audio_path + 'evalset=ljspeech+method=pinv+step=vocode+tho_type=fast___gomin___LJ002-0068.wav', sr=24000)[0]
    x_pinv_32k = librosa.load(plot_audio_path + 'evalset=ljspeech+method=pinv+step=vocode+tho_type=fast___gomin___LJ002-0068.wav', sr=32000)[0]

    x_oracle_24k = x_oracle_24k[:int(24000*1.36)]
    x_oracle_32k = x_oracle_32k[:int(32000*1.36)]
    x_diff_24k = x_diff_24k[:int(24000*1.36)]
    x_diff_32k = x_diff_32k[:int(32000*1.36)]
    x_pinv_24k = x_pinv_24k[:int(24000*1.36)]
    x_pinv_32k = x_pinv_32k[:int(32000*1.36)]

    mels_tr = bt.GominMelsTransform()
    tho_tr = bt.ThirdOctaveTransform(sr=32000, flen=4096, hlen=4000, refFreq=9, n_tho=20, db_delta=0)

    thirdo = tho_tr.wave_to_third_octave(x_oracle_32k)
    mels_oracle = mels_tr.wave_to_mels(x_oracle_24k)[0].detach().cpu().numpy().T
    mels_diff = mels_tr.wave_to_mels(x_diff_24k)[0].detach().cpu().numpy().T
    mels_pinv = mels_tr.wave_to_mels(x_pinv_24k)[0].detach().cpu().numpy().T

    # Calculate the size ratio for rows and columns
    thirdo = thirdo
    # thirdo = np.clip((thirdo + 100) / 100, 0.0, None)
    mels_oracle = mels_oracle * 100 -100
    mels_diff = mels_diff * 100 -100
    mels_pinv = mels_pinv * 100 -100
    # print(np.max(mels))
    # print(np.min(mels))
    print(np.max(thirdo))
    print(np.min(thirdo))
    # row_ratio = 80/29
    # col_ratio = 103/8
    # Interpolate along rows
    # thirdo = np.repeat(thirdo, row_ratio, axis=0)
    # thirdo = np.repeat(thirdo, col_ratio, axis=1)
    #thirdo = thirdo - (np.mean(thirdo)) + np.mean(x_mels_inf_pinv)

    # plot_raw_spectro(thirdo, vmin=0, vmax=1, ylabel='Third octave bin', name='raw_thirdo')
    # plot_raw_spectro(mels_oracle, vmin=0, vmax=1, ylabel='Third octave bin', name='raw_mels_oracle')
    # plot_raw_spectro(mels_diff, vmin=0, vmax=1, ylabel='Third octave bin', name='raw_mels_diff')
    # plot_raw_spectro(mels_pinv, vmin=0, vmax=1, ylabel='Third octave bin', name='raw_mels_pinv')

    # for noise_level in np.arange(0, 1.1, 0.1):
    #     mels_oracle_noisy = add_white_noise(mels_oracle, noise_level)
    #     mels_diff_noisy = add_white_noise(mels_diff, noise_level)
    #     plot_raw_spectro(mels_oracle_noisy, vmin=0, vmax=1, ylabel='Third octave bin', name=f'raw_mels_oracle_noise_{int(noise_level*10)}')
    #     plot_raw_spectro(mels_diff_noisy, vmin=0, vmax=1, ylabel='Third octave bin', name=f'raw_mels_diff_noise_{int(noise_level*10)}')

    # print(mels.shape)
    plot_multi_spectro([thirdo, mels_pinv, mels_diff, mels_oracle], 
             32000, 
            #  title=['Fast third-octaves', 'Mels (PINV)', 'Mels (transcoder)', 'Mels (original)'], 
             title=['Fast Third-Octave', 'Low-Res Mel (PINV)', 'High-Res Mel (Diffspec)', 'High-Res Mel (Original)'], 
             vmin=-100, 
             vmax=0,
             extlmax=1.23, 
             ylabel=['Third octave bin', 'Mel bin', 'Mel bin', 'Mel bin'], name="spectrograms", save=True, show_colorbar=False)