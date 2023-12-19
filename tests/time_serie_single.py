# In this file we try to learn a time serie using S4D

import lightning as L
import matplotlib.pyplot as plt
import torch

import actionq
from actionq.s4d import S4D

# TODO: make time series more difficult
xs = torch.linspace(0.0, 100.0, 1000)
ys = torch.sin(xs) + torch.cos(xs)

#plt.plot(ys)
#plt.show()

samples = []
targets = []

max_window_size = 100
min_window_size = 10

for ws in range(min_window_size, max_window_size + 1):
    for i in range(len(ys) - ws):
        sample = torch.zeros(ws)

        sample = ys[i:i+ws]
        target = ys[i+ws]

        samples.append(sample)
        targets.append(target)


