import torch
import lightning
import pandas as pd
import pickle
import os

from actionq.utils.transform import Compose

class SkeletonSubset():
    """Keep only a portion of the skeleton
    """
    
    def __init__(self, skeleton):
        self.skeleton = skeleton

    def __call__(self, sample):
        return sample[:, self.skeleton, :]

class JointRelativePosition():
    """Make all joint features relative to the root joints
    """

    def __init__(self, skeleton_roots_joints):
        self.root_joints = skeleton_roots_joints

    def __call__(self, sample):
        joint_root_mean = sample[:, self.root_joints, :].mean(axis=1, keepdims=True)
        return sample - joint_root_mean

class JointDifference():
    """Insert single step frame difference in the features
    """

    def __call__(self, sample):
        L, J, F = sample.shape
        result = torch.zeros((L-1, J, F * 2))
        result[..., :F] = sample[:-1, :, :]
        for frame in range(L-1):
            result[frame, :, F:] = sample[frame+1, :, :] - sample[frame, :, :]
        return result

class JointAngle2D():
    """Insert angle feature using x:f[0] and y:f[1]
    """

    def __init__(self, feat_x, feat_y):
        self.feat_x = feat_x
        self.feat_y = feat_y

    def __call__(self, sample):
        L, J, F = sample.shape
        result = torch.zeros((L, J, F + 1))
        result[..., :F] = sample
        result[..., -1] = torch.arctan(sample[..., self.feat_x] / (sample[..., self.feat_y] + 0.0001))
        return result

class RemovePosition():
    """Remove position features
    """

    def __call__(self, sample):
        L, J, F = sample.shape
        return sample[..., 2:F]

class CameraDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        window_size, 
        window_delta,
        transform=None,
        load_dir='data/processed/camera/0.1.5',
    ):
        super().__init__()
        self.max_score = -1

        # traverse all folders
        samples = []
        targets = []
        for session in os.scandir(load_dir):
            if os.path.isdir(session.path):
                for repetition in os.scandir(session.path):
                    
                    # load control factors to total score
                    total_score = 0
                    with open(os.path.join(repetition.path, 'control_factors.csv')) as f:
                        total_score = sum([ int(x) for x in f.readline().split(',') ])

                    self.max_score = max(self.max_score, total_score)

                    # load skeleton frames
                    with open(os.path.join(repetition.path, 'skeleton.pkl'), 'rb') as f:
                        skeleton = pickle.load(f)

                    frames = skeleton.shape[0]
                    if frames < window_size:
                        print(f'skeleton frames less than window size: {frames} < {window_size}')
                        continue

                    for beg in range(0, frames - window_size, window_delta):
                        sample = torch.from_numpy(skeleton[beg:beg+window_size, ...])
                        if transform is not None: 
                            sample = transform(sample)
                        targets.append(torch.tensor(total_score, dtype=torch.float32))
                        samples.append(sample)

        self.targets = torch.stack(targets, dim=0)
        self.samples = torch.stack(samples, dim=0)
        print(f'total samples: {self.samples.shape[0]}')

    # TODO: allow multi-gpu
    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]

    def __len__(self):
        return self.samples.shape[0]


class CameraDataModule(lightning.LightningDataModule):
    def __init__(self, batch_size, normalize, **kwargs):
        super().__init__()
        self.dataset = CameraDataset(**kwargs)
        self.batch_size = batch_size
        self.normalize = normalize

    def setup(self, stage: str = ''):

        if self.normalize:
            samples = self.dataset.samples
            mean = samples.mean(axis=(0, 1, 2), keepdims=True) # (1, 1, 1, F)
            std = samples.std(axis=(0, 1, 2), keepdims=True)
            samples = (samples - mean) / std
            print(f'dataset has been normalized\n\tmean = {mean}\n\tstd = {std}')

        # create validation set and train set
        self.train, self.val = torch.utils.data.random_split(
            self.dataset, [0.8, 0.2], torch.Generator())
    
    def features_count(self):
        return self.dataset[0][0].shape[-1]

    def max_score(self):
        return self.dataset.max_score

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train, 
            self.batch_size,
            drop_last=True,
            shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val, 
            self.batch_size,
        )

#if __name__ == '__main__':
#
#    # make root joints relative to skeleton subset
#    skeleton_root_joints = [UPPER_BODY_JOINTS.index(j) for j in UPPER_BODY_JOINTS_ROOTS]
#
#    dataset = CameraDataModule(
#        window_size=200, 
#        window_delta=50,
#        skeleton_joints=UPPER_BODY_JOINTS,
#        normalize=True,
#        batch_size=12,
#        transform=Compose([
#            JointRelativePosition(skeleton_root_joints),
#            # JointAngle2D(0, 1), ha qualche problema numerico
#            JointDifference()
#        ])
#    )
#    dataset.setup()