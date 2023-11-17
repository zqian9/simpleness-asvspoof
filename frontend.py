# -*- coding: utf-8 -*-
# @Author   : zqian9
import torch
from torch import nn
from torchaudio.transforms import Spectrogram


class LogSpectrogram(nn.Module):
    def __init__(self, n_fft=1024, win_length=400, hop_length=160):
        super().__init__()
        self.n_fft = n_fft
        self.window = torch.hann_window
        self.spectrogram = Spectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=2, onesided=True,
            window_fn=self.window, center=True, pad_mode='constant')

    def forward(self, x):
        """
        :param x: (batch_size, audio_length)
        :return: (batch_size, n_fft // 2, num_frames)
        """
        out = self.spectrogram(x)
        out = 10 * torch.log10(torch.clamp(out, min=1e-9))
        return out


# test
if __name__ == '__main__':
    model = LogSpectrogram()
    inp_ = torch.randn(1, 16000)
    out_ = model(inp_)
    print(out_.shape)
