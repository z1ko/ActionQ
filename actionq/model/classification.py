import torch
import torch.nn as nn
import lightning as L


class ActionClassifier(L.LightningModule):
    def __init__(self, model, lr, weight_decay, epochs=-1):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs

    def forward(self, samples):  # (B, L, J, F)
        logits = self.model(samples)  # (B, R)
        results = torch.nn.functional.sigmoid(logits)
        return results.squeeze()

    def training_step(self, batch, batch_idx):
        samples, _, targets = batch
        results = self.forward(samples)

        # print(f'results: {results}')
        # print(f'targets: {targets}')

        criterion = nn.BCELoss()
        # criterion = nn.MSELoss()
        loss = criterion(results, targets.float())
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        samples, _, targets = batch
        results = self.forward(samples)  # (B R)

        # print(f'outputs: {results}')
        # print(f'targets: {targets}')

        criterion = nn.BCELoss()
        loss = criterion(results, targets.float())

        prediction = (results > 0.5).long()
        correct = (prediction == targets).sum()
        size = prediction.size(0)

        acc = float(correct) / float(size) * 100.0
        self.log('val/accuracy', acc)
        self.log('val/loss', loss)

    # def test_step(self, batch, batch_idx):
    #    samples, targets = batch
    #    results = self.forward(samples)
    #    self.log_dict({
    #        'test-loss-mae': torch.nn.functional.l1_loss(results, targets),
    #        'test-loss-mse': torch.nn.functional.mse_loss(results, targets)
    #    })

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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)
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

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'test-crossentropy'
        }
