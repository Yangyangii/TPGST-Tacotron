import numpy as np
import pandas as pd
import os, sys
import torch
from torch.utils.data.dataset import Dataset
import glob, re
import utils
import codecs, unicodedata
from config import ConfigArgs as args
import jamo

class SpeechDataset(Dataset):
    """
    Basic Speech Dataset

    :param data_path: path to dataset
    :param metadata: path to metadata csv file
    :param mem_mode: Boolean. whether loads into memory

    """
    def __init__(self, data_path, metadata, mem_mode=False, training=True, training_ratio=0.99):
        self.data_path = data_path
        self.mem_mode = mem_mode
        meta = read_meta(os.path.join(data_path, metadata))
        n_rows = len(meta)
        np.random.seed(0)
        train_indices = np.random.choice(range(n_rows), int(n_rows*training_ratio), replace=False)
        
        meta = meta[~meta.index.isin(train_indices)] if not training else meta[meta.index.isin(train_indices)]
        self.fpaths, self.texts = [], []
        ch2idx, _ = load_vocab()
        meta.expanded = 'P' + meta.expanded + 'E'
        for fpath, text in zip(meta.fpath.values, meta.expanded.values):
            t = np.array([ch2idx[ch] for ch in jamo.h2j(text)])
            f = os.path.join(data_path, args.mel_dir, os.path.basename(fpath).replace('wav', 'npy'))
            self.texts.append(t)
            self.fpaths.append(f)
        if self.mem_mode:
            self.mels = [torch.tensor(np.load(os.path.join(
                self.data_path, args.mel_dir, path))) for path in self.fpaths]

    def __getitem__(self, idx):
        text, mel = None, None
        text = torch.tensor(self.texts[idx], dtype=torch.long)
        # Memory mode is faster
        if not self.mem_mode:
            mel = torch.tensor(np.load(self.fpaths[idx]))
        else:
            mel = self.mels[idx]
        mel = mel.view(-1, args.n_mels*args.r)
        return text, mel

    def __len__(self):
        return len(self.fpaths)

def load_vocab():
    """
    Makes dictionaries

    Returns:
        :char2idx: Dictionary containing characters as keys and corresponding indexes as values
        :idx2char: Dictionary containing indexes as keys and corresponding characters as values

    """
    char2idx = {char: idx for idx, char in enumerate(args.vocab)}
    idx2char = {idx: char for idx, char in enumerate(args.vocab)}
    return char2idx, idx2char

def text_normalize(text):
    """
    Normalizes text

    :param text: Text to be normalized

    Returns:
        text: Normalized text
    
    """
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                   if unicodedata.category(char) != 'Mn')  # Strip accents
    text = text.lower()
    text = re.sub(u"[^{}]".format(args.vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text

def read_meta(meta_path):
    # Parse
    meta = pd.read_table(meta_path, sep='|', header=None)
    meta.columns = ['fpath', 'ori', 'expanded', 'decomposed', 'duration', 'en']
    return meta

def collate_fn(data):
    """
    Creates mini-batch tensors from the list of tuples (texts, mels, mags, [spks]).

    :param data: list of tuple (texts, mels, mags, [spks]). each is a torch tensor of shape (B, Tx), (B, Ty/4, n_mels), (B, Ty, n_mags)

    Returns:
        :texts_pads: torch tensor of shape (batch_size, padded_length).
        :mels_pads: torch tensor of shape (batch_size, padded_length, n_mels).
        :mels_pads: torch tensor of shape (batch_size, padded_length, n_mags).
        :ff_pads: torch tensor of shape (batch_size, padded_langth, 1)
        :spks: torch tensor of shape (batch_size) 
    
    """
    # Sort a data list by text length (descending order).
    # data.sort(key=lambda x: len(x[0]), reverse=True)
    texts, mels = zip(*data)

    # Merge (from tuple of 1D tensor to 2D tensor).
    text_lengths = [len(text) for text in texts]
    mel_lengths = [len(mel) for mel in mels]
    # (number of mels, max_len, feature_dims)
    text_pads = torch.zeros(len(texts), max(text_lengths), dtype=torch.long)
    mel_pads = torch.zeros(len(mels), max(mel_lengths), mels[0].shape[-1])
    # final frame (N, Ty/r, 1)
    ff_pads = torch.zeros(len(mels), max(mel_lengths), 1)
    for idx in range(len(mels)):
        text_end = text_lengths[idx]
        text_pads[idx, :text_end] = texts[idx]
        mel_end = mel_lengths[idx]
        mel_pads[idx, :mel_end] = mels[idx]
        ff_pads[idx, mel_end-1:] = 1.0
    return text_pads, mel_pads, ff_pads

class TextDataset(Dataset):
    """
    Text Dataset for synthesis

    :param text path: String. path to text dataset
    :param ref_path: String. {<ref_path>, 'seen', 'unseen'}

    """
    def __init__(self, text_path, ref_path=None):
        self.texts = read_hangul(text_path)

    def __getitem__(self, idx):
        text = torch.tensor(self.texts[idx], dtype=torch.long)
        return text, None

    def __len__(self):
        return len(self.texts)

def read_text(path):
    """
    If we use pandas instead of this function, it may not cover quotes.

    :param path: String. metadata path

    Returns:
        texts: list of normalized texts

    """
    char2idx, _ = load_vocab()
    lines = codecs.open(path, 'r', 'utf-8').readlines()[1:]
    texts = []
    for line in lines:
        text = text_normalize(line.split(' ', 1)[-1]).strip() + u'E'  # ␃: EOS
        text = [char2idx[char] for char in text]
        texts.append(text)
    return texts

def read_hangul(path):
    """
    If we use pandas instead of this function, it may not cover quotes.

    :param path: String. metadata path

    Returns:
        texts: list of normalized texts

    """
    ch2idx, _ = load_vocab()
    lines = codecs.open(path, 'r', 'utf-8').readlines()
    texts = []
    for line in lines:
        hangul = 'P' + jamo.h2j(text_normalize(line).strip()) + 'E'
        t = np.array([ch2idx[ch] for ch in hangul])
        texts.append(t)
    return texts

def load_ref(path):
    """
    Load reference audios

    :param path: string. reference audio path

    Returns:
        refs: list of mel spectrograms from reference audios

    """
    fpaths = sorted(glob.glob(os.path.join(path, '*.wav')))
    refs = []
    for fpath in fpaths:
        mel, _ = utils.load_spectrogram(fpath)
        refs.append(mel)
    return refs

def synth_collate_fn(data):
    """
    Creates mini-batch tensors from the list of tuples (texts, mels, None, spks).

    :param data: list of tuple (texts, mels, mags, spks).

    Returns:
        text_pads: torch tensor of shape (batch_size, padded_length).
        mel_pads: torch tensor of shape (batch_size, padded_length, n_mels).
        spks: if spks is not none, torch tensor of shape (batch_size, padded_length) else none

    """
    # data.sort(key=lambda x: len(x[0]), reverse=True)
    texts, mels, _, spks = zip(*data)

    # Merge (from tuple of 1D tensor to 2D tensor).
    text_lengths = [len(text) for text in texts]
    mel_lengths = [len(mel) for mel in mels]
    # (number of mels, max_len, feature_dims)
    text_pads = torch.zeros(len(texts), max(text_lengths), dtype=torch.long)
    mel_pads = torch.zeros(len(mels), max(mel_lengths), mels[0].shape[-1])
    for idx in range(len(texts)):
        text_end = text_lengths[idx]
        text_pads[idx, :text_end] = texts[idx]
        mel_end = mel_lengths[idx]
        mel_pads[idx, :mel_end] = mels[idx]
    spks = torch.stack(spks, 0).squeeze(1) if spks is not None else None
    return text_pads, mel_pads, None, spks

def text_collate_fn(data):
    """
    Creates mini-batch tensors from the list of tuples (texts,).

    :param data: list of tuple (texts,).

    Returns:
        text_pads: torch tensor of shape (batch_size, padded_length).
    
    """
    # data.sort(key=lambda x: len(x[0]), reverse=True)
    texts, _ = zip(*data)

    # Merge (from tuple of 1D tensor to 2D tensor).
    text_lengths = [len(text) for text in texts]
    # (number of mels, max_len, feature_dims)
    text_pads = torch.zeros(len(texts), max(text_lengths), dtype=torch.long)
    for idx in range(len(texts)):
        text_end = text_lengths[idx]
        text_pads[idx, :text_end] = texts[idx]
    return text_pads, None, None
