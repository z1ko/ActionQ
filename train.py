
import os
import argparse
import pprint

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as L

from actionq.model.lru_model import LRUModel
from actionq.dataset.KiMoRe import KiMoReDataModule, skeleton_adj_matrix
from actionq.model.regression import ActionQ
# from actionq.model.s4 import AQS4
# from actionq.dataset.UIPRMD import UIPRMDDataModule
# from actionq.model.classification import ActionClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--window_size', type=int, default=200)
parser.add_argument('--window_delta', type=int, default=50)
parser.add_argument('--joint_count', type=int, help='number of joints in the skeleton')
parser.add_argument('--joint_features', type=int, help='features of each joint in the skeleton')
parser.add_argument('--joint_expansion', type=int, default=48)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--temporal_layers_count', type=int, default=6)
parser.add_argument('--spatial_layers_count', type=int, default=4)
parser.add_argument('--temporal_model', choices=['LRU', 'S4'], default='LRU')
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

# dataset = UIPRMDDataModule(
#    batch_size=16,
#    target_movement=1,
#    window_size=hparams['window_size'],
#    window_delta=hparams['window_delta'],
#    initial_frame_skip=hparams['initial_frame_skip']
# )
dataset.setup(task='regression')

train_dataloader = dataset.train_dataloader()
val_dataloader = dataset.val_dataloader()
# test_dataloader = dataset.test_dataloader()

# model = AQS4(
#    joint_features=3,
#    joint_count=19,
#    joint_expansion=hparams['joint_expansion'],
#    layers_count=hparams['layers_count'],
#    d_output=1  # Correct-incorrect probability
# )

model = LRUModel(
    joint_features=args.joint_features,
    joint_count=args.joint_count,
    joint_expansion=args.joint_expansion,
    temporal_layers_count=args.temporal_layers_count,
    spatial_layers_count=args.spatial_layers_count,
    output_dim=1,
    skeleton=skeleton_adj_matrix(),
    dropout=args.dropout,
    r_min=args.lru_min_radius,
    r_max=args.lru_max_radius
)

model = ActionQ(model, lr=args.learning_rate, maximum_score=50.0, weight_decay=0.001, epochs=args.epochs // 5)

logger = WandbLogger(project='ActionQ', save_dir='logs/')
logger.log_hyperparams(vars(args))

# Saves top 5 models
checkpoint_callback = ModelCheckpoint(
    save_top_k=5,
    monitor='validation/loss-mae',
    mode='min',
    dirpath='checkpoints/',
    filename='actionq-{epoch:03d}-{validation/loss-mae:.4f}',
    auto_insert_metric_name=False,
    every_n_epochs=50,
    verbose=True
)

trainer = L.Trainer(max_epochs=args.epochs, logger=logger, callbacks=[checkpoint_callback])
trainer.fit(
    model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader
)
