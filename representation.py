import lightning
import lightning.pytorch as pl
import argparse
import pprint

from actionq.model.representation import AQEncoder
from actionq.dataset.KiMoRe import KiMoReDataModule

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--window_size', type=int, default=200)
parser.add_argument('--window_delta', type=int, default=50)
parser.add_argument('--joint_count', type=int, help='number of joints in the skeleton')
parser.add_argument('--joint_features', type=int, help='features of each joint in the skeleton')
parser.add_argument('--joint_expansion', type=int, default=48)
parser.add_argument('--output_dim', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--temporal_layers_count', type=int, default=6)
parser.add_argument('--lru_min_radius', type=float, default=0.4)
parser.add_argument('--lru_max_radius', type=float, default=0.8)

args = parser.parse_args()
pprint.PrettyPrinter(indent=4).pprint(vars(args))

dataset = KiMoReDataModule(
    batch_size=args.batch_size,
    exercise=1,
    subjects=['expert', 'non-expert', 'stroke', 'backpain', 'parkinson'],
    window_size=args.window_size,
    window_delta=args.window_delta,
    features=['pos_x', 'pos_y', 'pos_z'],
    features_expansion=True,
    normalize=False
)

dataset.setup(task='regression')
train_dataloader = dataset.train_dataloader()
val_dataloader = dataset.val_dataloader()

# Model for representation learning
model = AQEncoder(
    input_dim=args.joint_features,
    joint_count=args.joint_count,
    state_dim=args.joint_expansion,
    temporal_layers_count=args.temporal_layers_count,
    output_dim=args.output_dim,
    dropout=args.dropout,
    r_min=args.lru_min_radius,
    r_max=args.lru_max_radius
)

# Output logs to web-ui
logger = pl.loggers.WandbLogger(project='ActionQ', save_dir='logs/')
logger.log_hyperparams(vars(args))

# Saves top 5 models
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    save_top_k=5,
    monitor='validation/loss-RnC',
    mode='min',
    dirpath='checkpoints/repr/',
    filename='actionq-repr-{epoch:03d}-{validation/loss-mae:.4f}',
    auto_insert_metric_name=False,
    every_n_epochs=10,
    verbose=True
)

trainer = lightning.Trainer(max_epochs=args.epochs, logger=logger, callbacks=[checkpoint_callback])
trainer.fit(
    model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader
)