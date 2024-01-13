import torch
from torch.utils.data import DataLoader, Dataset, random_split
import lightning as L
import matplotlib.animation
import matplotlib.pyplot as plt
import os
import random


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

_skeleton_joint_count = len(_skeleton_joint_names)


class KiMoReDataset(torch.utils.data.Dataset):
    """
        KInematic Assessment of MOvement and Clinical Scores for
        Remote Monitoring of Physical REhabilitation

        Each dataset item is a temporal skeleton evolution
        with a quality scores assigned
    """

    def __init__(self, root_dir, exercise, subjects):
        super().__init__()

        if any(s not in _subject_types for s in subjects):
            raise ValueError('Unrecognized subject type')

        if exercise not in _exercise_range:
            raise ValueError('Unrecognized exercise index')

        self.root_dir = root_dir
        self.subjects = subjects
        self.exercise = exercise

        self.max_frames_count = 0
        self.min_frames_count = 1e6

        self.samples = []
        self.targets = []

        self.dirs = [f'{self.root_dir}/{_subject_types[s]}' for s in subjects]
        for dir in self.dirs:
            self._load_all_from_directory(dir, self.exercise)

        print(f"LOG: loaded exercises samples count: {len(self.samples)}")
        print(f"LOG: max frames count: {self.max_frames_count}")
        print(f"LOG: min frames count: {self.min_frames_count}")

    def _load_all_from_directory(self, dir, exercise):
        for item in os.listdir(dir):
            folder = os.path.join(dir, item)
            if os.path.isdir(folder):
                self._load_single_sample(folder, exercise)

    def _load_single_sample(self, dir, exercise):
        base_dir = os.path.join(dir, f'Es{exercise}')
        print(f'LOG: Reading exercise folder of actor {base_dir}')

        # Load movement data
        raw_file = os.path.join(base_dir, 'Raw')
        for item in os.listdir(raw_file):
            if item.startswith('JointPosition'):
                with open(os.path.join(raw_file, item)) as f:

                    # Count Frames
                    frames = 0
                    for line in f.readlines():
                        if len(line) >= 25:
                            frames += 1

                    # Skip samples with too few frames
                    if frames < 400:
                        continue

                    self.max_frames_count = max(self.max_frames_count, frames)
                    self.min_frames_count = min(self.min_frames_count, frames)

                    # Use a fixed lenght for the samples.
                    # TODO: find a way to use different lenghts
                    frames = 400

                    print(f'LOG: Loading exercises with {frames} frames at {item}')
                    sample = torch.zeros((3, _skeleton_joint_count, frames))
                    # (Features, Joints, Frames)

                    t = 0
                    f.seek(0)
                    for line in f.readlines():
                        if t >= frames:
                            break
                        if len(line) >= 25:
                            self._parse_joint_pos_line(sample, line, t)
                            t += 1

                    # Rescale dataset
                    self._rescale_sample(sample)

                    self.samples.append(sample)
                    print(f"LOG: loaded sample: {sample}")

        # Load target data
        target_file = os.path.join(base_dir, 'Label')
        for item in os.listdir(target_file):
            if item.startswith('ClinicalAssessment') and item.endswith('.csv'):
                with open(os.path.join(target_file, item)) as f:
                    print(f'LOG: loading assessment for {target_file}')

                    line = f.readlines()[1].split(',')
                    target = torch.Tensor(3)  # TS PO CF
                    for i in range(3):
                        target[i] = float(line[1 + i * 5])

                    self.targets.append(target)
                    print(target)

    def _parse_joint_pos_line(self, sample, line, t):
        tokens = line.split(',')[:-1]
        if len(tokens) // 4 != 25:
            print(f'error in tokens lenght ({len(tokens)}): {tokens}')
            print(line)
            a = input()

        j = 0
        for i in range(0, 25):

            # Skip some unimportant joints (feet, hands)
            if i in [15, 19, 21, 22, 23, 24]:
                continue

            sample[0, j, t] = float(tokens[j * 4 + 0])
            sample[1, j, t] = float(tokens[j * 4 + 1])
            sample[2, j, t] = float(tokens[j * 4 + 2])
            j += 1

    def _rescale_sample(self, sample):
        mean_x = torch.mean(sample[0, :, :])
        mean_y = torch.mean(sample[1, :, :])

        sample[0, :, :] -= mean_x
        sample[1, :, :] -= mean_y

        max_x = torch.max(sample)
        min_x = torch.min(sample)
        delta = max_x + min_x

        print(f"LOG: rescaling [{min_x}, {max_x}] -> [-1, 1]")
        sample = (sample - min_x) / delta * 2.0 - 1.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]


class KiMoReDataModule(L.LightningDataModule):
    """
        Dataloader for the KiMoRe dataset
    """

    def __init__(self, root_dir, batch_size, exercise, subjects):
        super().__init__()
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.exercise = exercise
        self.subjects = subjects

    def setup(self, _stage: str = ''):
        self.total = KiMoReDataset(self.root_dir, self.exercise, self.subjects)
        self.train, self.val, self.test = torch.utils.data.random_split(
            self.total, [0.8, 0.1, 0.1], torch.Generator())

        print(f'LOG: total samples count: {len(self.total)}')
        print(f'LOG: train samples count: {len(self.train)}')
        print(f'LOG: val   samples count: {len(self.val)}')
        print(f'LOG: test  samples count: {len(self.test)}')

    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, self.batch_size)


class KiMoReDataVisualizer:

    def visualize_2d(self, item):

        # Skeleton edges
        # edges = [
        #    [0, 1], [1, 18], [2, 18], [2, 3], [4, 18], [8, 18],
        #    [4, 5], [8, 9], [0, 12], [0, 16], [12, 13], [16, 17]
        # ]

        sample, _ = item
        frames = sample.size(-1)

        fig = plt.figure()
        axs = fig.add_subplot(111)
        plt.xlim(-1.0, 1.0)
        plt.ylim(-1.0, 1.0)

        xs = sample[0, :, :]  # torch.rand((10, frames))
        ys = sample[1, :, :]  # torch.rand((10, frames))

        # Draw bones 2d
        # bones = []
        # for edge in edges:
        #    a, b = edge
        #    bone, = axs.plot([xs[a, 0], xs[b, 0]], [ys[a, 0], ys[b, 0]])
        #    bones.append(bone)

        # Draw joints 2d
        graph, = axs.plot(xs[:, 0], ys[:, 0], linestyle="", marker="o")

        def update(frame):
            graph.set_data(xs[:, frame], ys[:, frame])
            axs.set_title(f'frame: {frame}/{frames}')
            return graph,

        a = matplotlib.animation.FuncAnimation(fig, update,
                                               frames=frames, interval=30)
        plt.show()

    def visualize_time_series(self, item):
        sample, _ = item

        coords = ['x', 'y']
        _, axs = plt.subplots(5, (_skeleton_joint_count + 5) // 6)
        for joint, ax in enumerate(axs.flat):
            if joint < 19:
                joint_name = _skeleton_joint_names[joint]
                ax.set_title(joint_name)
                for i in range(2):
                    ax.plot(sample[i, joint, :])

                ax.legend(coords, loc='upper right')

        plt.tight_layout()
        plt.show()
