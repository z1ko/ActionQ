import torch
import lightning as L

class ActionQ(L.LightningModule):
    def __init__(self, model, lr, weight_decay, epochs=-1):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs

    def forward(self, samples):  # (B, L, J, F)
        #print(samples.shape)
        return self.model(samples)

    def training_step(self, batch, batch_idx):
        samples, targets = batch 
        results = self.forward(samples)
        loss = torch.nn.functional.l1_loss(results, targets)
        self.log('train-loss-mae', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        samples, targets = batch
        results = self.forward(samples)

        mad_loss = torch.nn.functional.l1_loss(results, targets)
        self.log('validation-loss-mae', mad_loss)

        rmse_loss = torch.sqrt(torch.nn.functional.mse_loss(results, targets))
        self.log('validation-loss-rmse', rmse_loss)

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
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.2)
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
            'lr_scheduler': scheduler,
            #'monitor': 'train-loss-mae'
        } 
