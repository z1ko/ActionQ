# In this file we try to learn a time serie using S4D

import lightning as L
import matplotlib.pyplot as plt
import torch

from ..actionq.model.actionq import S4Model

# TODO: make time-serie more difficult


class TimeserieDataset(torch.utils.data.Dataset):
    def __init__(self, window_size):
        super().__init__()
        xs = torch.linspace(0.0, 1000.0, 10000)
        self.ys = torch.sin(xs) + torch.cos(xs * 0.3) * 0.6
        self.ws = window_size

        # plt.plot(self.ys)
        # plt.show()

    def __len__(self):
        return len(self.ys) - self.ws

    def __getitem__(self, idx):
        sample = self.ys[idx:idx + self.ws]
        sample.unsqueeze_(-1)
        target = self.ys[idx + self.ws]
        return sample, target


class TimeserieDataModule(L.LightningDataModule):
    def __init__(self, window_size, batch_size):
        super().__init__()
        self.window_size = window_size
        self.batch_size = batch_size

    def setup(self, _stage: str = ''):
        self.dataset_total = TimeserieDataset(self.window_size)
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(
            self.dataset_total,
            [0.8, 0.2],
            torch.Generator()
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset_train, self.batch_size, num_workers=15)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset_val, self.batch_size, num_workers=15)


# data = TimeserieDataModule(20, 10)
# data.setup()
#
# logger = TensorBoardLogger('logs')
#
# model = S4Model(d_input=1, d_output=1, d_model=256, n_layers=4)
# model = ActionQ(model, 0.001, 0.01, 20)
# print(model)
#
# trainer = L.Trainer(max_epochs=20, logger=logger)
# trainer.fit(model, train_dataloaders=data.train_dataloader(), val_dataloaders=data.val_dataloader())
