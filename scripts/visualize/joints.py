
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import pandas as pd
import numpy as np
import torch
import sys

# Hack assurdo per caricare moduli da altri cartelle
sys.path.append('.')
from actionq.dataset.KiMoRe import rescale_sample

# Name of all skeleton joints
SKELETON_JOINT_NAMES = [
    'spine_base',
    'spine_mid',
    'neck',
    'head',
    'shoulder_left',
    'elbow_left',
    'wrist_left',
    'hand_left',
    'shoulder_right',
    'elbow_right',
    'wrist_right',
    'hand_right',
    'hip_left',
    'knee_left',
    'ankle_left',
    'hip_right',
    'knee_right',
    'ankle_right',
    'spine_shoulder'
]

parser = argparse.ArgumentParser()

parser.add_argument('-s', '--subject')
parser.add_argument('-e', '--exercise')
parser.add_argument('-m', '--mode', 
                    choices=['series', 'animation'])

args = parser.parse_args()

# ===================================================================================
# Style configuration

mpl.style.use('seaborn-v0_8-deep')
mpl.rcParams['figure.figsize'] = (20, 5)
mpl.rcParams['figure.dpi'] = 100

# ===================================================================================
# Load data

samples_df = pd.read_parquet('data/processed/kimore_samples.parquet.gzip')
samples_df = samples_df.query(f"subject == '{args.subject}'").query(f"exercise == {args.exercise}")

sample = samples_df[['frame', 'joint', 'pos_x', 'pos_y', 'pos_z']]
sample.set_index(['frame', 'joint'], inplace=True)

frames_count = len(sample.index.get_level_values(0).unique())
joints_count = len(sample.index.get_level_values(1).unique())

sample = np.reshape(sample, (frames_count, joints_count, 3))
sample = torch.tensor(sample) # (F, J, 3)
#rescale_sample(sample)

# ===================================================================================
# Plot single subject, single exercise joint's features evolution

if args.mode == 'series':

    def plot_joint(df, joint, ax):
        
        ax.plot(sample[:, joint, 0], label='pos_x') # pos_x
        ax.plot(sample[:, joint, 1], label='pos_y') # pos_y
        ax.plot(sample[:, joint, 2], label='pos_z') # pos_z

        ax.set_title(f'joint = {SKELETON_JOINT_NAMES[joint]}')
        ax.set_ylabel('value')
        ax.set_xlabel('frames')
        ax.legend()

    fig, ax = plt.subplots(nrows=(joints_count + 4 - 1) // 4, ncols=4, sharex=True, figsize=(20, 10))
    fig.suptitle(f'subject = {args.subject}, exercise = {args.exercise}', fontsize=16)
    
    fax = ax.flatten()
    fax[-1].axis('off')
    for joint in range(joints_count):
        plot_joint(sample, joint, fax[joint])

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.show()

if args.mode == 'animation':
    fig, ax = plt.subplots()
    
    xs, ys = sample[:, :, 0], sample[:, :, 1]
    ax.set_xlim(torch.min(xs), torch.max(xs))
    ax.set_ylim(torch.min(ys), torch.max(ys))

    graph, = ax.plot(xs[:, 0], ys[:, 0], linestyle="", marker="o")

    def update(frame):
        graph.set_data(xs[frame, :], ys[frame, :])
        ax.set_title(f'frame: {frame}/{frames_count}')
        return graph,

    _ = anim.FuncAnimation(fig, update, frames=frames_count, interval=30)
    plt.show()
