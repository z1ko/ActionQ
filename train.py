
import os
import argparse
import pprint

import lightning as L
from lightning.pytorch.loggers import WandbLogger

from actionq.rnn.lru import LRUModel
from actionq.model.s4 import AQS4
from actionq.dataset.KiMoRe import KiMoReDataModule
from actionq.dataset.UIPRMD import UIPRMDDataModule
from actionq.model.regression import ActionQ
from actionq.model.classification import ActionClassifier

parser = argparse.ArgumentParser()
parser.add_argument('-ep', '--epochs', type=int, default=100)
parser.add_argument('-lr', '--learning_rate', default=0.001)
parser.add_argument('-bs', '--batch_size', default=10)
parser.add_argument('-ws', '--window_size', default=200)
parser.add_argument('-lc', '--layers_count', default=4)
parser.add_argument('-je', '--joint_expansion', default=6)
parser.add_argument('-do', '--dropout', default=0.25)
parser.add_argument('-tm', '--temporal_model', choices=['LRU', 'S4'], default='LRU')

args = parser.parse_args()
pprint.PrettyPrinter(indent=4).pprint(vars(args))

logger = WandbLogger(project='ActionQ', save_dir='logs/')
logger.log_hyperparams(vars(args))

dataset = KiMoReDataModule(
    batch_size=args.batch_size,
    exercise=4,
    subjects=['expert', 'non-expert', 'stroke', 'backpain', 'parkinson'],
    window_size=args.window_size,
    features=['pos_x', 'pos_y', 'pos_z'],
    features_expansion=True
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
#test_dataloader = dataset.test_dataloader()

# model = AQS4(
#    joint_features=3,
#    joint_count=19,
#    joint_expansion=hparams['joint_expansion'],
#    layers_count=hparams['layers_count'],
#    d_output=1  # Correct-incorrect probability
# )

model = LRUModel(
    joint_features=3,
    joint_count=19,
    joint_expansion=args.joint_expansion,
    layers_count=args.layers_count,
    output_dim=1,
    dropout=args.dropout
)

model = ActionQ(model, lr=args.learning_rate, maximum_score=50.0, weight_decay=0.001, epochs=100)
trainer = L.Trainer(max_epochs=args.epochs, logger=logger)
trainer.fit(
    model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader
)

# trainer.test(
#    dataloaders=train_dataloader
# )
