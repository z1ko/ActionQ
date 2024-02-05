import torch
import torch.nn as nn
import lightning as L

class ActionQS4(L.LightningModule):
    def __init__(self, lr, weight_decay):
        super().__init__()



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

        criterion = nn.HuberLoss(reduction='mean', delta=1.0)
        loss = criterion(y_model, y_target)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        samples, targets = batch
        results = self.forward(samples)
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
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
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
