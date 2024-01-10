from torch.utils.data import DataLoader
from actionq.dataset.KiMoRe import KiMoReDataModule, KiMoReDataVisualizer
#
dataset = KiMoReDataModule(
    root_dir='data/KiMoRe',
    batch_size=1,
    subjects=['expert'],
    exercise=1
)

dataset.setup()
dataloader = DataLoader(dataset.dataset_total)

visualizer = KiMoReDataVisualizer()
for batch in dataloader:
    sample = batch[0]
    visualizer.visualize_2d(sample)

exit(0)

import torch
import torch.functional as F
import matplotlib.pyplot as plt
from actionq.model.s4 import S4Model
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger


class ActionQ(L.LightningModule):
    def __init__(self, model, lr, weight_decay, epochs):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs

    def training_step(self, batch, batch_idx):
        samples, targets = batch
        results = self.model(samples)
        results.squeeze_()

        loss = torch.nn.functional.l1_loss(results, targets)
        self.log('loss/l1/train', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        samples, targets = batch
        results = self.model(samples)
        results.squeeze_()

        loss = torch.nn.functional.l1_loss(results, targets)
        self.log('loss/l1/val', loss)

    def test_step(self, batch, batch_idx):
        samples, targets = batch
        results = self.model(samples)
        results.squeeze_()

        loss = torch.nn.functional.l1_loss(results, targets)
        self.log('loss/l1/test', loss)

    def configure_optimizers(self):
        all_parameters = list(model.parameters())

        # General parameters don't contain the special _optim key
        params = [p for p in all_parameters if not hasattr(p, "_optim")]

        # Create an optimizer with the general parameters
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
        hps = [
            dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
        ]  # Unique dicts
        for hp in hps:
            params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group(
                {"params": params, **hp}
            )

        # Create a lr scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

        # Print optimizer info
        # keys = sorted(set([k for hp in hps for k in hp.keys()]))
        # for i, g in enumerate(optimizer.param_groups):
        #    group_hps = {k: g.get(k, None) for k in keys}
        #    print(' | '.join([
        #        f"Optimizer group {i}",
        #        f"{len(g['params'])} tensors",
        #    ] + [f"{k} {v}" for k, v in group_hps.items()]))

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }


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

#data = TimeserieDataModule(20, 10)
#data.setup()
#
#logger = TensorBoardLogger('logs')
#
#model = S4Model(d_input=1, d_output=1, d_model=256, n_layers=4)
#model = ActionQ(model, 0.001, 0.01, 20)
#print(model)
#
#
#trainer = L.Trainer(max_epochs=20, logger=logger)
#trainer.fit(model, train_dataloaders=data.train_dataloader(), val_dataloaders=data.val_dataloader())


# sup
