import numpy as np
import argparse
import matplotlib.pyplot as plt
import einops as ein
import pprint
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('--normalize', default=False, action='store_true')
parser.add_argument('--joint', type=int, default=15)

args = parser.parse_args()
pprint.pprint(vars(args))

# Più semplice di così...
with open(args.input, 'rb') as f:
    skeleton = pickle.load(f)

L, J, F = skeleton.shape

# Derived features
velocity = np.zeros((L-1, J, 2))
for frame in range(L-1):
    velocity[frame, :, :] = skeleton[frame+1, :, :] - skeleton[frame, :, :]

skeleton = np.concatenate([skeleton[:-1, ...], velocity], axis=-1)

if args.normalize:
    L, J, F = skeleton.shape
    skeleton = ein.rearrange(skeleton, 'L J F -> L (J F)')
    
    mean = np.mean(skeleton, axis=0)
    sd = np.std(skeleton, axis=0)
    skeleton = (skeleton - mean) / sd

    skeleton = ein.rearrange(skeleton, 'L (J F) -> L J F', L=L, J=J)

plt.plot(skeleton[:, args.joint, :])
plt.legend(['pos_x', 'pos_y', 'vel_x', 'vel_y'])
plt.show()

