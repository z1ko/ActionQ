from lightning.pytorch.loggers import TensorBoardLogger
import lightning as L
from actionq.model.s4 import AQS4
import matplotlib.pyplot as plt
import torch.functional as F
import torch
from torch.utils.data import DataLoader
from actionq.dataset.KiMoRe import KiMoReDataModule, KiMoReDataVisualizer, KiMoReDataset2
from einops import rearrange


D = KiMoReDataset2(exercise=1, rescale_samples=True)

#visualizer = KiMoReDataVisualizer()
#for i, sample in enumerate(D):
#    visualizer.visualize_2d(sample)
#
#exit(0)

dataset = KiMoReDataModule(
    root_dir='data/KiMoRe',
    batch_size=8,
    subjects=['expert', 'non-expert', 'stroke', 'parkinson', 'backpain'],
    exercise=1
)

dataset.setup()

#
# visualizer = KiMoReDataVisualizer()
# for batch in dataloader:
#    sample = batch[0]
#    visualizer.visualize_2d(sample)
#

train_dataloader = dataset.train_dataloader()
val_dataloader = dataset.val_dataloader()
test_dataloader = dataset.test_dataloader()


class ActionQ(L.LightningModule):
    def __init__(self, model, lr, weight_decay, epochs=-1):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs

    def forward(self, samples):  # (B, F, J, L)
        samples = samples.permute(0, -1, -2, -3)  # (B, L, J, F)
        return self.model(samples)

    def training_step(self, batch, batch_idx):
        samples, targets = batch  # samples: (B, F, J, L)
        results = self.forward(samples)
        loss = torch.nn.functional.l1_loss(results, targets)
        self.log('loss/l1/train', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        samples, targets = batch  # samples: (B, F, J, L)
        results = self.forward(samples)

        mad_loss = torch.nn.functional.l1_loss(results, targets)
        self.log('loss/l1/val', mad_loss)

        rmse_loss = torch.sqrt(torch.nn.functional.mse_loss(results, targets))
        self.log('loss/rmse/val', rmse_loss)

    # def test_step(self, batch, batch_idx):
    #    samples, targets = batch
    #    results = self.model(samples)
    #    results.squeeze_()
    #
    #    loss = torch.nn.functional.l1_loss(results, targets)
    #    self.log('loss/l1/test', loss)

    def configure_optimizers(self):
        all_parameters = list(self.model.parameters())

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
        keys = sorted(set([k for hp in hps for k in hp.keys()]))
        for i, g in enumerate(optimizer.param_groups):
            group_hps = {k: g.get(k, None) for k in keys}
            print(' | '.join([
                f"Optimizer group {i}",
                f"{len(g['params'])} tensors",
            ] + [f"{k} {v}" for k, v in group_hps.items()]))

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }


logger = TensorBoardLogger(save_dir='./logs')

model = AQS4(joint_features=3, joint_count=19, joint_expansion=32, layers_count=4, d_output=3)
model = ActionQ(model, 0.001, 0.01)

trainer = L.Trainer(max_epochs=400, logger=logger)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
