import torch
import torch.nn as nn
import lightning
import einops as ein

from actionq.loss.rank_n_contrast import RnCLoss
from actionq.rnn.lru import LRULayer, TemporalAggregator

class LRUSkeletonEncoder(nn.Module):
    def __init__(
        self,
        joint_count,
        input_dim,
        state_dim,
        output_dim,
        temporal_layers_count,
        temporal_agg_method,
        dropout,
        **kwargs
    ):
        super().__init__()

        # Convert joint features to state dim
        self.initial = nn.Sequential(
            nn.Linear(input_dim, state_dim),
            nn.LeakyReLU(),
            nn.Linear(state_dim)
        )

        # Convert state dim to output dim
        self.final = nn.Sequential(
            nn.Linear(state_dim * joint_count, output_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(output_dim * 2, output_dim)
        )

        # Temporal layers
        self.temporal_layers = nn.ModuleList()
        for _ in range(temporal_layers_count):
            self.temporal_layers.append(
                LRULayer(
                    state_dim=state_dim,
                    activation='gelu',
                    dropout=dropout,
                    **kwargs
                )
            )

        # How to aggregate in time
        self.temporal_agg = TemporalAggregator(method=temporal_agg_method)

    def forward(self, x): # (B, L, J, F)

        B, L, F = x.shape
        x = ein.rearrange(x, 'B L J F -> (B J) L F')

        x = self.initial(x)
        for layer in self.temporal_layers:
            x = layer(x)

        x = self.temporal_agg(x)
        x = self.final(x)

        y = ein.rearrange('(B J) F -> B J F', B=B, J=J)
        return y


class AQEncoder(lightning.LightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.temporal_model = LRUSkeletonEncoder(**kwargs)
        self.criterion = RnCLoss(temperature=2.0, label_diff='l1', feature_sim='l2')
        self.save_hyperparameters()

    # TODO: Add spatial model
    def forward(self, x):
        return self.temporal_model(x)

    def training_step(self, batch, _):
        samples, labels = batch
        features = self.forward(samples)
        loss = self.criterion(features, labels)
        self.log('train/loss-RnC', loss)
        return loss
    
    def validation_step(self, batch, _):
        samples, labels = batch
        features = self.forward(samples)
        loss = self.criterion(features, labels)
        self.log('validation/loss-RnC', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=self.scheduler_step, gamma=0.1)
        return { 'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'train/loss-RnC' }