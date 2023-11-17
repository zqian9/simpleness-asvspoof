# -*- coding: utf-8 -*-
# @Author   : zqian9
import os
import librosa

import torch
from torch.utils.data import Dataset

import config


def gen_files_meta(data_root, dataset_type='train', track='LA'):
    dict_meta = {}
    cm_protocols_root = os.path.join(data_root, "{}2019/ASVspoof2019_{}_cm_protocols".format(track, track))

    if dataset_type == 'train':
        txt_path = os.path.join(
            cm_protocols_root, "ASVspoof2019.{}.cm.train.trn.txt".format(track))
    else:
        txt_path = os.path.join(
            cm_protocols_root, "ASVspoof2019.{}.cm.{}.trl.txt".format(track, dataset_type))

    with open(txt_path, 'r') as f:
        line_meta = f.readlines()
    for line in line_meta:
        _, name, _, category, label = line.strip().split(' ')
        category = 0 if category == '-' else int(category[1:])
        label = 0 if label == 'bonafide' else 1
        dict_meta[name] = [label, category]
    return dict_meta


def gen_file_names(data_root, track='LA'):
    filenames = []
    txt_path = os.path.join(
        data_root, "{}2021/ASVspoof2021_{}_eval/ASVspoof2021.{}.cm.eval.trl.txt".format(track, track, track))
    with open(txt_path, 'r') as f:
        line_meta = f.readlines()
    for line in line_meta:
        name = line.strip()
        filenames.append(name)
    return filenames


def pad_sequence(x, max_len=64000):
    # need to pad
    num_repeats = int(max_len / len(x)) + 1
    padded_x = x.repeat(num_repeats)
    return padded_x[:max_len]


class ASVSpoof19(Dataset):
    def __init__(self, data_root, duration, dataset_type, track='LA', sample_rate=16000):
        """
        :param data_root:
        :param duration:
        :param dataset_type: 'train', 'dev', 'eval'
        :param track:
        :param sample_rate:
        """
        self.data_root = data_root
        self.duration = duration
        self.dataset_type = dataset_type
        self.track = track
        self.sample_rate = sample_rate

        self.meta = gen_files_meta(data_root, dataset_type, track)
        self.filenames = list(self.meta.keys())

    def __getitem__(self, index):
        filename = self.filenames[index]
        label = self.meta[filename][0]
        category = self.meta[filename][1]

        wave_path = os.path.join(
            self.data_root, "{}2019/ASVspoof2019_{}_{}/flac/{}.flac".format(
                self.track, self.track, self.dataset_type, filename
            )
        )
        waveform, _ = librosa.load(wave_path, sr=self.sample_rate)
        if len(waveform) >= int(self.duration * self.sample_rate):
            waveform = waveform[:int(self.duration * self.sample_rate)]
        else:
            waveform = pad_sequence(waveform, int(self.duration * self.sample_rate))
        waveform = torch.from_numpy(waveform)

        return waveform, label, category, filename

    def __len__(self):
        return len(self.filenames)


class ASVSpoof21(Dataset):
    def __init__(self, data_root, duration, track='LA', sample_rate=16000):
        """
        :param data_root:
        :param duration:
        :param track:
        :param sample_rate:
        """
        self.data_root = data_root
        self.duration = duration
        self.track = track
        self.sample_rate = sample_rate

        self.filenames = gen_file_names(data_root, track)

    def __getitem__(self, index):
        filename = self.filenames[index]

        wave_path = os.path.join(
            self.data_root, "{}2021/ASVspoof2021_{}_eval/flac/{}.flac".format(
                self.track, self.track, filename
            )
        )
        waveform, _ = librosa.load(wave_path, sr=self.sample_rate)
        if len(waveform) >= int(self.duration * self.sample_rate):
            waveform = waveform[:int(self.duration * self.sample_rate)]
        else:
            waveform = pad_sequence(waveform, int(self.duration * self.sample_rate))
        waveform = torch.from_numpy(waveform)
        waveform = torch.clamp(waveform, min=-1.0, max=1.0)

        return waveform, filename

    def __len__(self):
        return len(self.filenames)


# test example
if __name__ == '__main__':
    train_dataset = ASVSpoof19(
        config.data_root, duration=4,
        dataset_type='train')
    for x, y, c, f in train_dataset:
        print(x.shape, y, c, f)

