import random

import numpy as np
import torch
import torch.nn as nn

_CONV_KERNEL_SIZE = 3
_CONV_STRIDE = 1
_CONV_PADDING = 1
_DEFAULT_ARCH = ((2, 64), (2, 128), (2, 256), (2, 512))
_IN_CHANNELS = 1
_OUT_CHANNELS = 1
_RANDOM_STATE = 42


class _NConv(nn.Module):
    def __init__(self, N, in_channels, out_channels):
        super(_NConv, self).__init__()
        self._check_n(N)
        self.convs = nn.Sequential()
        for ii in range(N):
            self.convs.extend((
                nn.Conv2d(
                    in_channels=in_channels if ii == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=_CONV_KERNEL_SIZE,
                    stride=_CONV_STRIDE,
                    padding=_CONV_PADDING,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=out_channels),
                nn.ReLU(inplace=True),
            ))

    def _check_n(self, n):
        if n <= 0:
            raise ValueError("_NConv.N must be positive")
        else:
            return True

    def forward(self, x):
        return self.convs(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=_IN_CHANNELS,
        out_channels=_OUT_CHANNELS,
        arch=_DEFAULT_ARCH,
        random_state=_RANDOM_STATE,
    ):
        super(UNet, self).__init__()
        self._check_arch(arch)
        self.dconvs = nn.ModuleList()
        self.uconvs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self._add_down_convs(in_channels, arch)
        self._add_up_convs(arch)
        self._add_bottleneck(arch)
        self._add_final_conv(out_channels, arch)
        self._set_seed(random_state)

    def _add_down_convs(self, in_channels, arch):
        for nconvs, nfeatures in arch:
            self.dconvs.append(_NConv(nconvs, in_channels, nfeatures))
            in_channels = nfeatures

    def _add_up_convs(self, arch):
        for nconvs, nfeatures in reversed(arch):
            self.uconvs.extend((
                nn.ConvTranspose2d(nfeatures * 2, nfeatures, kernel_size=2, stride=2),
                _NConv(nconvs, nfeatures * 2, nfeatures),
            ))

    def _add_bottleneck(self, arch):
        nconvs, nfeatures = arch[-1]
        self.bootleneck = _NConv(nconvs, nfeatures, nfeatures * 2)

    def _add_final_conv(self, out_channels, arch):
        _, nfeatures = arch[0]
        self.finalconv = nn.Conv2d(nfeatures, out_channels, kernel_size=1)

    def _check_arch(self, arch):
        if not isinstance(arch, (list, tuple)):
            raise ValueError(
                "UNet.arch is expected to be an iterable of pairs of integers, for"
                f" example, ((2, 64), (3, 128)), but got {arch}"
            )
        kk = 2 ** len(arch)
        for ii, step in enumerate(arch):
            if not isinstance(step, (list, tuple)):
                raise ValueError(f"UNet.arch[{ii}] must be an interable")
            elif len(step) != 2:
                raise ValueError(f"UNet.arch[{ii}] must be of size 2")
            for jj in range(2):
                if not isinstance(step[jj], int):
                    raise ValueError(f"UNet.arch[{ii}][{jj}] must be an integer")
                elif step[jj] <= 0:
                    raise ValueError(f"UNet.arch[{ii}][{jj}] must be positive")
            if step[1] % kk != 0:  # ensure deconv works correctly
                raise ValueError(f"UNet.arch[{ii}][1] must be divisible by {kk}")
        return True

    def _set_seed(self, seed=_RANDOM_STATE):
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)

    def forward(self, x):
        skips = []
        for downconv in self.dconvs:
            x = downconv(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bootleneck(x)
        skips = skips[::-1]
        for ii, upconv in enumerate(self.uconvs):
            if ii % 2 == 0:  # upsampling
                x = upconv(x)
                skip = skips[ii // 2].to(x.device)
                x = torch.cat((skip, x), dim=1)
            else:
                x = upconv(x)
        x = self.finalconv(x)
        return x


class UNetNorm(UNet):
    def forward(self, x):
        x = super().forward(x)
        x = torch.sigmoid(x)  # scale between [0, 1]
        return x
