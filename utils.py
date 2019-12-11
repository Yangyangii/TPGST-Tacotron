from config import ConfigArgs as args
import librosa
import numpy as np
import os, sys
from scipy import signal
import copy
import torch
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

def load_spectrogram(fpath):
    """
    Load spectrograms from file path

    :param fpath: file path to spectrogram

    Returns:
        :mel: (Ty/r, n_mels*r) numpy array
        :mag: (Ty, 1+n_fft//2) numpy array
        
    """
    wav, sr = librosa.load(fpath, sr=args.sr)

    wav = wav / np.abs(wav).max() * 0.999
    ## Pre-processing
    wav, _ = librosa.effects.trim(wav)
    # STFT
    linear = librosa.stft(y=wav,
                          n_fft=args.n_fft,
                          hop_length=args.hop_length,
                          win_length=args.win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(args.sr, args.n_fft, args.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    mel = np.log(np.clip(mel, 1e-5, 1e+5))
    maxval, minval = 3.0, np.log(1e-5)
    mel = (mel - minval) / (maxval - minval)

    mag = np.log(np.clip(mag, 1e-5, 1e+5))
    maxval, minval = 6.0, np.log(1e-5)
    mag = (mag - minval) / (maxval - minval)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    mel, mag = padding_reduction(mel, mag)
    return mel, mag

def padding_reduction(mel, mag):
    """
    Pads for reduction factor

    :param mel: (T, n_mels) numpy array
    :param mag: (T, 1+n_fft//2) numpy array

    Returns:
        :mel: (Ty/r, n_mels*r) numpy array
        :mag: (Ty, 1+n_fft//2) numpy array

    """
    # Padding
    t = mel.shape[0]
    n_paddings = args.r - (t % args.r) if t % args.r != 0 else 0  # for reduction
    mel = np.reshape(np.pad(mel, [[0, n_paddings], [0, 0]], mode="constant"), [-1, args.n_mels*args.r])
    mag = np.pad(mag, [[0, n_paddings], [0, 0]], mode="constant")
    # mel = mel[::args.r, :] # DCTTS
    return mel, mag

def att2img(A):
    """
    To normalize attention ***

    :param A: (1, Tx, Ty) Tensor

    Returns:
        :A: (1, Tx, Ty) Tensor
    
    """
    for i in range(A.shape[-1]):
        att = A[0, :, i]
        local_min, local_max = att.min(), att.max()
        A[0, :, i] = (att-local_min)/(local_max-local_min)
    return A


def plot_att(A, text, global_step, path='.', name=None):
    """
    Saves attention

    :param A: (Tx, Ty) numpy array
    :param text: (Tx,) list
    :param global_step: scalar
    :param path: String. path to save attention
    :param name: String. Save attention as this name

    """
    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(A)
    fig.colorbar(im, fraction=0.035, pad=0.02)
    fig.suptitle('{} Steps'.format(global_step), fontsize=32)
    plt.xlabel('Text', fontsize=28)
    plt.ylabel('Time', fontsize=28)
    # plt.xticks(np.arange(len(text)), text)
    if name is not None:
        plt.savefig(os.path.join(path, name), format='png')
    else:
        plt.savefig(os.path.join(path, 'A-{}.png'.format(global_step)), format='png')
    plt.close(fig)

def lr_policy(step):
    """
    warm up learning rate function

    :param step:

    Returns:
        :updated learning rate: scalar.
    """
    return args.warm_up_steps**0.5 * np.minimum((step+1) * args.warm_up_steps**-1.5, (step+1)**-0.5)