import torch
import lightning
import pickle
import os


class UIPRMDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        target_movement,
        window_size,
        window_delta,
        use_all_movements=False,  # If true the all the other movements will be used as 'incorrect' data
        initial_frame_skip=None,
        filename='uiprmd.pkl'
    ):

        filepath = os.path.join('data/processed/', filename)
        with open(filepath, 'rb') as f:
            self.data = pickle.load(f)

        # process all samples
        self.samples = []
        for category_id, category in self.data.items():
            for movement_id, movement in category.items():

                # Skip other movements
                if not use_all_movements and movement_id != target_movement:
                    continue

                for subject_id, complete_sample in movement.items():
                    for beg in range(0, complete_sample.shape[0] - window_size + 1, window_delta):
                        if initial_frame_skip is not None and beg < initial_frame_skip:
                            continue  # NOTE: Skip first frames, they are probably problematic

                        # print(f'loading {category_id}/{movement_id}/{subject_id} in [{beg}, {beg+window_size}]')
                        sample = torch.tensor(complete_sample[beg:beg + window_size, :, :], dtype=torch.float32)
                        is_target = category_id == 'correct' and movement_id == target_movement
                        metadata = (category_id, movement_id, subject_id)
                        self.samples.append((sample, metadata, 1.0 if is_target else 0.0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class UIPRMDDataModule(lightning.LightningDataModule):
    def __init__(self, batch_size=16, **kwargs):
        self.batch_size = batch_size
        self.kwargs = kwargs

    def setup(self, _stage=None):
        uiprmd_full = UIPRMDDataset(**self.kwargs)
        self.uiprmd_train, self.uiprmd_val, self.uiprmd_test = torch.utils.data.random_split(
            uiprmd_full, [0.8, 0.2, 0.0]
        )

    def calculate_weights(self, data):
        target_count = 0
        for i in range(0, len(data)):
            _, _, is_target = data[i]
            if is_target != 0.0:
                target_count += 1

        not_target_count = len(data) - target_count
        weight_not_target = 1.0 / not_target_count
        weight_target = 1.0 / target_count

        weights = torch.zeros(len(data))
        for i in range(0, len(data)):
            _, _, is_target = data[i]
            if is_target != 0.0:
                weights[i] = weight_target
            else:
                weights[i] = weight_not_target

        return weights

    def train_dataloader(self):
        # TODO: Understand how to treat imbalanced data classes
        # weights = self.calculate_weights(self.uiprmd_train)
        # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(self.uiprmd_train), replacement=True)

        return torch.utils.data.DataLoader(
            self.uiprmd_train,
            batch_size=self.batch_size,
            # sampler=sampler,
            # shuffle=False
            shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.uiprmd_val,
            batch_size=self.batch_size,
            shuffle=False
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.uiprmd_test,
            batch_size=self.batch_size,
            shuffle=False
        )


if __name__ == '__main__':
    # dataset = UIPRMDDataset(target_movement=1, window_size=200, window_delta=50, frame_skip=100)
    # print(len(dataset))

    datamodule = UIPRMDDataModule(
        batch_size=16,
        target_movement=1,
        window_size=200,
        window_delta=50,
        initial_frame_skip=100
    )

    datamodule.setup()
    print('len: ', len(datamodule.uiprmd_train))

    for item in datamodule.train_dataloader():
        sample, metadata, target = item

        targets = torch.sum(target != 0.0)
        print(targets / datamodule.batch_size)
