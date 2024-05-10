import torch
from torch.utils.data import DataLoader, Dataset, random_split
import lightning as L
import matplotlib.animation
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

# All available subjects in the dataset
_subject_types = {
    'expert': 'CG/Expert',
    'non-expert': 'CG/NotExpert',
    'stroke': 'GPP/Stroke',
    'parkinson': 'GPP/Parkinson',
    'backpain': 'GPP/BackPain'
}

# All available exercise types in the dataset
_exercise_range = range(0, 5)

# Name of all skeleton joints
_skeleton_joint_names = [
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

# Name of all skeleton joints
_skeleton_joint_ids = {
    'spine_base': 0,
    'spine_mid': 1,
    'neck': 2,
    'head': 3,
    'shoulder_left': 4,
    'elbow_left': 5,
    'wrist_left': 6,
    'hand_left': 7,
    'shoulder_right': 8,
    'elbow_right': 9,
    'wrist_right': 10,
    'hand_right': 11,
    'hip_left': 12,
    'knee_left': 13,
    'ankle_left': 14,
    'hip_right': 15,
    'knee_right': 16,
    'ankle_right': 17,
    'spine_shoulder': 18
}

_skeleton_connections = {
    'spine_base': ['spine_mid', 'hip_left', 'hip_right'],
    'spine_mid': ['spine_base', 'spine_shoulder'],
    'neck': ['head', 'spine_shoulder'],
    'head': ['neck'],
    'shoulder_left': ['spine_shoulder', 'elbow_left'],
    'elbow_left': ['shoulder_left', 'wrist_left'],
    'wrist_left': ['elbow_left', 'hand_left'],
    'hand_left': ['wrist_left'],
    'shoulder_right': ['spine_shoulder', 'elbow_right'],
    'elbow_right': ['shoulder_right', 'wrist_right'],
    'wrist_right': ['elbow_right', 'hand_right'],
    'hand_right': ['wrist_right'],
    'hip_left': ['spine_base', 'knee_left'],
    'knee_left': ['hip_left', 'ankle_left'],
    'ankle_left': ['knee_left'],
    'hip_right': ['spine_base', 'knee_right'],
    'knee_right': ['hip_left', 'ankle_right'],
    'ankle_right': ['knee_right'],
    'spine_shoulder': ['neck', 'shoulder_left', 'shoulder_right', 'spine_mid']
}


def skeleton_adj_matrix():
    joints_count = len(_skeleton_joint_names)
    result = torch.zeros((joints_count, joints_count))
    for joint_name, joint_connections in _skeleton_connections.items():
        src = _skeleton_joint_ids[joint_name]
        result[src, src] = 1.0  # self loop
        for dst in map(lambda j: _skeleton_joint_ids[j], joint_connections):
            result[src, dst] = 1.0
            result[dst, src] = 1.0
    return result


def rescale_sample(sample):
    # TODO: This is hardcoded to 2 dimensions only
    mean_x = torch.mean(sample[:, :, 0])
    mean_y = torch.mean(sample[:, :, 1])

    sample[:, :, 0] -= mean_x
    sample[:, :, 1] -= mean_y

    max_x = torch.max(sample)
    min_x = torch.min(sample)
    delta = max_x + min_x

    # print(f"LOG: rescaling [{min_x}, {max_x}] -> [-1, 1]")
    sample = (sample - min_x) / delta * 2.0 - 1.0


# class KiMoReDatasetClassification(torch.utils.data.Dataset):
#    def __init__(self, features, window_size, rescale_samples=True):
#        super().__init__()
#        self.samples = []
#
#        samples_path = 'data/processed/kimore_samples.parquet.gzip'
#        df = pd.read_parquet(samples_path)
#
#        # One-hot encoding of exercise
#        exercise_encoding = {
#            1: torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32),
#            2: torch.tensor([0, 1, 0, 0, 0], dtype=torch.float32),
#            3: torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32),
#            4: torch.tensor([0, 0, 0, 1, 0], dtype=torch.float32),
#            5: torch.tensor([0, 0, 0, 0, 1], dtype=torch.float32)
#        }
#
#        for exercise, exercise_data in df.groupby('exercise'):
#            for subject, subject_data in exercise_data.groupby('subject'):
#
#                subject_data.set_index(['frame', 'joint'], inplace=True)
#                subject_data = subject_data[features]
#
#                # Use index to obtain tensor dimensionality
#                frames_count = len(subject_data.index.get_level_values(0).unique())
#                joints_count = len(subject_data.index.get_level_values(1).unique())
#
#                # Skip samples that are too short
#                if frames_count < window_size:
#                    print(f'WARNING: sample too small: {subject}, frames: {frames_count}')
#                    continue
#
#                try:
#                    sample_complete = torch.tensor(subject_data.to_numpy(), dtype=torch.float32)
#                    sample_complete = sample_complete.reshape(frames_count, joints_count, len(features))
#
#                except Exception as e:
#                    # TODO: Understand the failure cases
#                    print(f'WARNING for subject: {subject}, frames: {frames_count}\n', e)
#                    continue
#
#                # Create a sample for each window
#                for frame_begin in range(0, frames_count - window_size, window_size):
#                    sample = sample_complete[frame_begin:frame_begin + window_size, :, :]
#                    if rescale_samples:
#                        rescale_sample(sample)
#
#                    self.samples.append((sample, exercise_encoding[exercise]))
#
#    def __len__(self):
#        return len(self.samples)
#
#    def __getitem__(self, idx):
#        return self.samples[idx]

class KiMoReDataset(torch.utils.data.Dataset):
    """
        KInematic Assessment of MOvement and Clinical Scores for
        Remote Monitoring of Physical REhabilitation

        Each dataset item is a temporal skeleton evolution
        with a quality scores assigned

        Each sample is of shape (frames, joints, features)
    """

    def __init__(
        self,
        exercise,
        subjects,
        features,
        window_size,
        window_delta,
        features_expansion,
        normalize
    ):
        super().__init__()

        if exercise not in _exercise_range:
            raise ValueError(f'Exercise {exercise} not in range {_exercise_range}')

        self.features = features
        self.samples = []

        # Load targets from processed dataset
        targets_path = 'data/processed/kimore_targets.parquet.gzip'
        df = pd.read_parquet(targets_path)
        self.targets_df = df[(df['exercise'] == exercise)]

        # Load only specified subjects
        if subjects is not None:
            if any(s not in _subject_types.keys() for s in subjects):
                raise ValueError(f'subjects {subjects} not in {_subject_types.keys()}')

            filter = [_subject_types[s] for s in subjects]
            self.targets_df = self.targets_df[self.targets_df['type'].isin(filter)]

        subject_target_map = {}
        for subject, target in self.targets_df.groupby('subject'):
            target = target['TS'].to_numpy()  # Total Clinical Score
            subject_target_map[subject] = torch.tensor(target, dtype=torch.float32)

        # Load samples from processed dataset
        samples_path = 'data/processed/kimore_samples.parquet.gzip'
        df = pd.read_parquet(samples_path)
        self.samples_df = df[df['exercise'] == exercise]

        # Transform dataframe to tensor samples
        for name, subject in self.samples_df.groupby('subject'):

            if name not in subject_target_map:
                print(f'WARNING: target not found for subject {name}')
                continue
            target = subject_target_map[name]

            subject.set_index(['frame', 'joint'], inplace=True)
            subject = subject[self.features]

            # Use index to obtain tensor dimensionality
            frames_count = len(subject.index.get_level_values(0).unique())
            joints_count = len(subject.index.get_level_values(1).unique())

            # Skip samples that are too short
            if frames_count < window_size:
                print(f'WARNING: sample too small: {name}, frames: {frames_count}')
                continue

            try:
                sample_all = torch.tensor(subject.to_numpy(), dtype=torch.float32)
                sample_all = sample_all.reshape(frames_count, joints_count, len(self.features))

            except Exception as e:
                # TODO: Understand the failure cases
                print(f'WARNING for subject: {name}, frames: {frames_count}\n', e)
                continue

            # Create a sample for each window
            for frame_begin in range(0, frames_count - window_size, window_delta):
                # print(f'LOG: creating sample [{frame_begin}-{frame_begin+window_size}]')
                sample = sample_all[frame_begin:frame_begin + window_size, :, :]
                self.samples.append((sample, target))

        if features_expansion:
            self.samples = self.feature_augmentation(self.samples)

    def feature_augmentation(self, samples):
        results = []
        for movement, target in samples:
            L, J, F = movement.shape
            augmented = torch.zeros((L - 1, J, F + 3))
            augmented[:, :, :3] = movement[:-1, ...]

            for frame in range(L - 1):
                augmented[frame, :, 3:6] = movement[frame + 1, :, :] - movement[frame, :, :]  # first difference

            results.append((augmented, target))
        return results

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class KiMoReDataModule(L.LightningDataModule):
    """
        Dataloader for the KiMoRe dataset
    """

    def __init__(self, batch_size, **dataset_args):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_args = dataset_args

    def setup(self, task='classification'):
        if task == 'classification':
            self.dataset = None  # KiMoReDatasetClassification(**self.dataset_args)
        else:
            self.dataset = KiMoReDataset(**self.dataset_args)

        # TODO: Implement k-fold cross validation
        self.train, self.val, self.test = torch.utils.data.random_split(
            self.dataset, [0.8, 0.2, 0.0], torch.Generator())

        print(f'LOG: total samples count: {len(self.dataset)}')
        print(f'LOG: train samples count: {len(self.train)}')
        print(f'LOG: val   samples count: {len(self.val)}')
        # print(f'LOG: test  samples count: {len(self.test)}')

    def train_dataloader(self):
        return DataLoader(
            self.train,
            self.batch_size,
            drop_last=True,
            drop_last=True,
            shuffle=True
        )

    def val_dataloader(self):
        # TODO: Validation is evaluated on a single batch,
        # maybe it is better to set the batch_size to the
        # size of the entire validation set.
        return DataLoader(
            self.val,
            self.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            self.batch_size,
        )
