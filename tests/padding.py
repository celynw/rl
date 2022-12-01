#!/usr/bin/env python3
import torch

from rl.models.utils import Decay3d

n_input_channels = 2
t2d = torch.rand(1, n_input_channels, 84, 84)
t3d = torch.rand(1, n_input_channels, 6, 84, 84)

conv2d = torch.nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0)
conv3d = torch.nn.Conv3d(n_input_channels, 32, kernel_size=(1, 8, 8), stride=(1, 4, 4), padding=(0, 0, 0))
decay3d = Decay3d(n_input_channels, 32, kernel_size=8, stride=4, padding=0)

o2d = conv2d(t2d)
o3d = conv3d(t3d)
od3d = conv3d(t3d)

print(f"t2d.shape: {t2d.shape}")
print(f"t3d.shape: {t3d.shape}")
print(f"o2d.shape: {o2d.shape}")
print(f"o3d.shape: {o3d.shape}")
print(f"od3d.shape: {od3d.shape}")

n_input_channels = 2

conv2d = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
conv3d = torch.nn.Conv3d(32, 64, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 0, 0))
decay3d = Decay3d(32, 64, kernel_size=4, stride=2, padding=0)

o2d = conv2d(o2d)
o3d = conv3d(o3d)
od3d = conv3d(od3d)

print(f"o2d.shape: {o2d.shape}")
print(f"o3d.shape: {o3d.shape}")
print(f"od3d.shape: {od3d.shape}")
