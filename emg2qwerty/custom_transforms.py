"""
Custom transformation and augmentation code for CSE247A class project, Neil Jones.
"""
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_wavelets import DWT1DForward


@dataclass
class RandomScaledBandRotation:
    """Applies band rotation augmentation by shifting the electrode channels
    by an offset value chosen uniformly randomly between ``offsets``.
    The output is scaled such that if a band is rotated by 0.25 (for example)
    then the resulting tensor will be 0.75 of the signal from the unrotated
    sensor, plus 0.25 of the signal from the rotated.

    NOTE: If the input is 3D with batch dim (TNC), then this transform
    applies band rotation for all items in the batch with the same offset.
    To apply different rotations each batch item, use the ``ForEach`` wrapper.

    Args:
        offsets (list): List of integers denoting the offsets by which the
            electrodes are allowed to be shift. A random offset from this
            list is chosen for each application of the transform.
        channel_dim (int): The electrode channel dimension. (default: -1)
    """

    offsets: Sequence[int] = (-1, 1)
    channel_dim: int = -1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        offset = np.random.uniform(low=self.offsets[0], high=self.offsets[1]) if len(self.offsets) > 0 else 0
        if offset < 0:
            rotated = tensor.roll(-1, dims=self.channel_dim)
        elif offset > 0:
            rotated = tensor.roll(+1, dims=self.channel_dim)
        else:
            rotated = tensor

        return (1-np.abs(offset)) * tensor + np.abs(offset) * rotated


@dataclass
class LogWaveletSpectrogram:
    """Creates log10-scaled wavelet decomposition from an EMG signal.
    In the case of multi-channeled signal, the channels are treated
    independently.  The input must be of shape (T, ...) and the
    returned spectrogram is of shape (T, ..., freq).

    This is probably not super fast, but maybe it will be fast enough
    to test out.

    Args:
        wavelet (str): the wavelet basis to use (must be DWT-compatible).
            (default: haar)
        num_levels_out (int): the number of scales to use in the output to the
            wavelet decomposition, to mimic n_fft.
            (default: 33)
        num_levels (int): the number of actual levels to use in the decomposition.
            (default: 10)
        time_scale (int): factor to dilate the time dimension for downsampling, to
            mimic hop_length. (default: 4)
    """

    wavelet: str = 'haar'
    num_levels_out: int = 33
    time_scale: int = 16
    num_levels: int = 10
    log_normalize: bool = True

    def __post_init__(self) -> None:
        self.dwt = DWT1DForward(J=self.num_levels, wave=self.wavelet, mode='zero')

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        #print(f'Tensor shape={tensor.shape}')
        assert self.time_scale > 0, 'Time scale must be larger than 0.'
        T = tensor.shape[0]
        C = tensor.shape[-1]
        time_bins = 622  # T // self.time_scale

        #def _band(t, band):
        #    # We throw away Y1, the low-frequency coefficients, as we claim without
        #    # any kind of evidence that they are not useful. Maybe we can go back
        #    # later and change that?
        #    _, Yh = self.dwt(t[:,band,:,:])
        #    stack = []
        #    for i in range(len(Yh)):
        #        stack.append(F.interpolate(Yh[i], size=(time_bins,),
        #                                   mode='linear', align_corners=True))
        #    stacked = torch.stack(stack).movedim(-1, 0).movedim(1, -1)
        #    # A DWT with 32 levels is idiotic, since the levels are spaced at powers
        #    # of 2. This takes a more reasonable number of levels and does linear
        #    # interpolation to get to the number of bins out that we expect.
        #    stacked = F.interpolate(stacked, size=(C, self.num_levels_out),
        #                            mode='bilinear', align_corners=True)
        #    return stacked

        #print(f'tensor.shape={tensor.shape}')
        x = tensor.movedim(0, -1)  # (T, ..., C) -> (..., C, T)
        #print(f'x.shape={x.shape}')
        _, Yh = self.dwt(x)
        stack = []
        for i in range(len(Yh)):
            stack.append(F.interpolate(Yh[i], size=(time_bins,),
                                       mode='linear', align_corners=True))
        stacked = torch.stack(stack)
        #print(f'stacked.shape={stacked.shape}')
        stacked = stacked.movedim(-1, 0).movedim(1, -1)
        #print(f'post swap stacked.shape={stacked.shape}')
        # A DWT with 32 levels is idiotic, since the levels are spaced at powers
        # of 2. This takes a more reasonable number of levels and does linear
        # interpolation to get to the number of bins out that we expect.
        stacked = F.interpolate(stacked, size=(C, self.num_levels_out),
                                mode='bilinear', align_corners=True)
        #print(f'post interp stacked.shape={stacked.shape}')
        if self.log_normalize:
            logspec = torch.log10(torch.abs(stacked) + 1e-6)
        else:
            logspec = stacked
        return logspec  # (T, ..., C, freq)
