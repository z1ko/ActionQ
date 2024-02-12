import argparse
import pprint
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

UPPER_BODY_JOINTS='11,12,13,14,15,16'
UPPER_BODY_JOINTS_ROOTS='11,12'
AUGMENTATIONS='angle,difference'

parser = argparse.ArgumentParser()
parser.add_argument('input_folder', type=str),
parser.add_argument('output', type=str)

parser.add_argument('--joint_subset', default=UPPER_BODY_JOINTS)
parser.add_argument('--joint_roots', default=UPPER_BODY_JOINTS_ROOTS)
parser.add_argument('--make_relative_to_roots', default=True, action='store_true')
parser.add_argument('--augmentations', default=AUGMENTATIONS)
parser.add_argument('--plot_along', default=False, action='store_true')

# TODO: This argument should be part of the dataset loader, not the preprocessor
#parser.add_argument('--normalize', default=True, action='store_true')

args = parser.parse_args()
pprint.pprint(vars(args))

def aug_difference(samples):
    S, L, J, F = samples.shape
    result = np.zeros((S, L-1, J, 2 * F))
    result[:, :, :, :F] = samples[:, :-1, :, :]
    for frame in range(L-1):
        result[:, frame, :, F:] = samples[:, frame+1, :, :] - samples[:, frame, :, :]
    
    return result

def aug_angle(samples):
    S, L, J, F = samples.shape
    result = np.zeros((S, L, J, F + 1))
    result[:, :, :, :F] = samples
    ratios = np.expand_dims(samples[..., 0] / samples[..., 1], axis=-1)
    result[:, :, :, F:] = np.arctan(ratios)
    return result

# Supported augmentations
aug_fn = {
    'difference': aug_difference,
    'angle': aug_angle
}

def plot_all_joints_of_sample(samples, sample_idx):
    C, R = 2, (samples.shape[2]  + 2 - 1) // 2
    fig, axs = plt.subplots(nrows=R, ncols=C)

    sample = samples[sample_idx]
    for i, ax in zip(range(sample.shape[1]), axs.flatten()):
        for f in range(sample.shape[-1]):
            ax.plot(sample[:, i, f], label=f'joint-{i}-feat{f}')
        ax.legend()

    plt.tight_layout()
    plt.show()

def plot_none(samples, sample_idx):
    pass

# Plot only if requested
plot_sample = plot_all_joints_of_sample if args.plot_along else plot_none

def import_skeleton_data(filepath):
    with open(filepath, 'rb') as f:
        skeleton = pickle.load(f)
        print(f'loaded skeleton: {skeleton.shape}')
        return skeleton

skeletons = []
for folder in os.scandir(args.input_folder):
    if os.path.isdir(folder.path):
        for file in os.scandir(folder):
            if file.name.endswith('skeleton.pkl'):
                skeleton = import_skeleton_data(file.path)
                skeletons.append(skeleton)

# find smallest sample
min_frames = min(map(lambda s: s.shape[0], skeletons))
print(f'minimum number of frames: {min_frames}')

_, J, F = skeletons[0].shape
samples = np.zeros((len(skeletons), min_frames, J, F))
for i, skeleton in enumerate(skeletons):
    samples[i, ...] = skeleton[:min_frames, ...]

# Extract only desired joints
if args.joint_subset is not None:
    joint_subset = list(map(int, args.joint_subset.split(',')))
    samples = samples[:, :, joint_subset, :]

plot_sample(samples, 0)

# Insert augmented features
if args.augmentations is not None:
    for aug in args.augmentations.split(','):
        print(f'Augmenting feature: {aug}')
        samples = aug_fn[aug](samples)
        plot_sample(samples, 0)

# Make all body joints position relative to root ones if requested
if args.make_relative_to_roots and args.joint_roots is not None:
    joint_roots = list(map(lambda j: joint_subset.index(int(j)), args.joint_roots.split(',')))
    joint_root_mean = samples[:, :, joint_roots, :].mean(axis=(1,2), keepdims=True)
    samples = samples - joint_root_mean

    plot_sample(samples, 0)

# Normalize all data if requested
#if args.normalize:
#    global_mean = samples.mean(axis=(0,1,2), keepdims=True)
#    global_std = samples.std(axis=(0,1,2), keepdims=True)
#    samples = (samples - global_mean) / global_std
    
# Save preprocessed data to file
with open(args.output, 'wb') as f:
    pickle.dump(samples, f)