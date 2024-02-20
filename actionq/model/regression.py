from typing import Any
import torch
import torch.nn as nn
import lightning as L

from actionq.loss.rank_n_contrast import RnCLoss
from actionq.model.lru_model import LRUModel


class ActionQ(L.LightningModule):
    def __init__(self, model, lr, weight_decay, maximum_score, epochs=-1):
        super().__init__()
        self.model = model
        self.lr = lr
        self.maximum_score = maximum_score
        self.weight_decay = weight_decay
        self.epochs = epochs

        self.save_hyperparameters(ignore=['model'])

    def forward(self, samples):  # (B, L, J, F)
        scores = self.model(samples)
        return scores * self.maximum_score  # Maximum score in the dataset

    def training_step(self, batch, batch_idx):
        samples, y_target = batch
        y_model = self.forward(samples)

        # Tanto per...
        # mse_loss = torch.nn.functional.mse_loss(results, targets)
        # mae_loss = torch.nn.functional.l1_loss(results, targets)

        criterion = nn.HuberLoss(reduction='mean', delta=1.35)
        loss = criterion(y_model.squeeze(-1), y_target)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        samples, targets = batch
        results = self.forward(samples).squeeze(-1)
        self.log_dict({
            'validation/loss-mae': torch.nn.functional.l1_loss(results, targets),
            'validation/loss-mse': torch.nn.functional.mse_loss(results, targets)
        }, prog_bar=True)

    def test_step(self, batch, batch_idx):
        samples, targets = batch
        results = self.forward(samples)
        self.log_dict({
            'test/loss-mae': torch.nn.functional.l1_loss(results, targets),
            'test/loss-mse': torch.nn.functional.mse_loss(results, targets)
        }, prog_bar=True)

    def predict_step(self, frame, state):  # (J, F)
        return self.model.forward_with_state(frame, state) * self.maximum_score

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
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #    optimizer,
        #    mode='min',
        #    patience=20,
        #    factor=0.2,
        #    verbose=True
        # )

        # Print optimizer info
        # keys = sorted(set([k for hp in hps for k in hp.keys()]))
        # for i, g in enumerate(optimizer.param_groups):
        #    group_hps = {k: g.get(k, None) for k in keys}
        #    print(' | '.join([
        #        f"Optimizer group {i}",
        #        f"{len(g['params'])} tensors",
        #    ] + [f"{k} {v}" for k, v in group_hps.items()]))

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=400, gamma=0.1)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'train-loss-mse'
        }

    @staticmethod
    def add_model_specific_args(root_parser):
        parser = root_parser.add_argument_group('AQS4')
        parser.add_argument('--test')
        return parser


class ActionQualityModule(L.LightningModule):
    def __init__(self, lr, weight_decay, scheduler_step) -> None:
        super().__init__()
        self.weight_decay = weight_decay
        self.scheduler_step = scheduler_step
        self.lr = lr

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=self.scheduler_step, gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}


class AQEncoder(ActionQualityModule):
    def __init__(self, embedding_dim, options, **kwargs) -> None:
        super().__init__(**kwargs)

        self.criterion = RnCLoss(temperature=2, label_diff='l1', feature_sim='l2')
        self.model = LRUModel(
            joint_features=options.joint_features,
            joint_count=options.joint_count,
            joint_expansion=options.joint_expansion,
            temporal_layers_count=options.temporal_layers_count,
            spatial_layers_count=options.spatial_layers_count,
            output_dim=1,
            # skeleton=skeleton_adj_matrix(),
            dropout=options.dropout,
            r_min=options.lru_min_radius,
            r_max=options.lru_max_radius
        )

        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(root_parser):
        parser = root_parser.add_argument_group('Encoder')
        parser.add_argument('--embedding_dim', type=int, default=128)
        return parser

    def training_step(self, batch, batch_idx):
        samples, targets = batch
        outputs = self.model(samples)
        return self.criterion(outputs, targets)


class AQRegressor(ActionQualityModule):
    def __init__(self, embedding_dim) -> None:
        super().__init__()

        self.criterion = nn.HuberLoss(delta=1.35)
        self.model = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def training_step(self, batch, batch_idx):
        samples, targets = batch
        outputs = self.model(samples)
        return self.criterion(outputs, targets)
