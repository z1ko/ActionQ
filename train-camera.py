import torch
import lightning
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pprint
import datetime
import einops as ein

from actionq.dataset.camera import CameraDataModule, JointRelativePosition, JointDifference, SkeletonSubset, RemovePosition
from actionq.utils.transform import Compose
from actionq.model.regression import ActionQ
from actionq.model.lru_model import LRUModel

UPPER_BODY_JOINTS=[11,12,13,14,15,16]
UPPER_BODY_JOINTS_ROOTS=[11,12]

parser = argparse.ArgumentParser()

dataset_opts = parser.add_argument_group('dataset')
dataset_opts.add_argument('--window_size', type=int, default=200)
dataset_opts.add_argument('--window_delta', type=int, default=50)
dataset_opts.add_argument('--batch_size', type=int, default=12)
dataset_opts.add_argument('--dataset', type=str)

lru_opts = parser.add_argument_group('linear-recurrent-unit')
lru_opts.add_argument('--lru_min_radius', type=float, default=0.4)
lru_opts.add_argument('--lru_max_radius', type=float, default=0.8)

model_opts = parser.add_argument_group('model')
model_opts.add_argument('--joint_expansion', type=int, default=256)
model_opts.add_argument('--temporal_layers_count', type=int, default=1)
model_opts.add_argument('--dropout', type=float, default=0.1)

optimizer_opts = parser.add_argument_group('optimizer')
optimizer_opts.add_argument('--learning_rate', type=float, default=0.0001)
optimizer_opts.add_argument('--epochs', type=int, default=800)

opts = parser.parse_args()

# makes root joints relative to skeleton subset
skeleton_root_joints = [UPPER_BODY_JOINTS.index(j) for j in UPPER_BODY_JOINTS_ROOTS]
dataset = CameraDataModule(
    load_dir=opts.dataset,
    window_size=opts.window_size, 
    window_delta=opts.window_delta,
    normalize=True,
    batch_size=opts.batch_size,
    transform=Compose([
        SkeletonSubset(UPPER_BODY_JOINTS),
        JointRelativePosition(skeleton_root_joints),
        JointDifference()
    ])
)

dataset.setup()
train = dataset.train_dataloader()
val = dataset.val_dataloader()

# Main model
model = ActionQ(
    LRUModel(
        joint_features=dataset.features_count(),
        joint_count=len(UPPER_BODY_JOINTS),
        joint_expansion=opts.joint_expansion,
        temporal_layers_count=opts.temporal_layers_count,
        spatial_layers_count=0,
        output_dim=1,
        dropout=opts.dropout,
        r_min=opts.lru_min_radius,
        r_max=opts.lru_max_radius
    ), 
    lr=opts.learning_rate, 
    maximum_score=dataset.max_score(), 
    weight_decay=0.001, 
    epochs=opts.epochs // 5
)

# Saves top 5 models
now = datetime.datetime.now()
checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(
    save_top_k=5,
    monitor='validation/loss-mae',
    mode='min',
    dirpath=f'checkpoints/camera/{now}/',
    filename='actionq-{epoch:03d}-{validation/loss-mae:.4f}',
    auto_insert_metric_name=False,
    every_n_epochs=50,
    verbose=True
)

# Logging to web-ui
logger = lightning.pytorch.loggers.WandbLogger(project='ActionQ-Camera', save_dir='logs/')
logger.log_hyperparams(vars(opts))

trainer = lightning.Trainer(max_epochs=opts.epochs, logger=logger, callbacks=[checkpoint_callback])
trainer.fit(
    model,
    train_dataloaders=train,
    val_dataloaders=val
)