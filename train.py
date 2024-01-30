
import os
import argparse
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from actionq.model.s4 import AQS4
from actionq.dataset.KiMoRe import KiMoReDataModule
from actionq.dataset.UIPRMD import UIPRMDDataModule
from actionq.model.regression import ActionQ
from actionq.model.classification import ActionClassifier

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=100, type=int)
parser.add_argument('-lr', '--learning_rate', default=0.001)
args = parser.parse_args()

logger = WandbLogger(
    project='AQS4',
    save_dir='logs/'
)

# Experiment configuration
hparams = {
    # Dimension of a batch
    'batch_size': 10,
    # Number of frames for each sample
    'window_size': 200,
    # Offset between each window of frames
    'window_delta': 50,
    # Frames skipped at the beginning
    'initial_frame_skip': 100,
    # How many layers of S4 are used to model the time sequence
    'layers_count': 6,
    # Expansion of each joint features
    'joint_expansion': 3
}
logger.log_hyperparams(hparams)


dataset = KiMoReDataModule(
   batch_size=hparams['batch_size'],
   exercise=1,
   subjects=['expert', 'non-expert', 'stroke'],
   window_size=hparams['window_size'],
   features=['pos_x', 'pos_y', 'pos_z'],
   features_expansion=True
)

#dataset = UIPRMDDataModule(
#    batch_size=16,
#    target_movement=1,
#    window_size=hparams['window_size'],
#    window_delta=hparams['window_delta'],
#    initial_frame_skip=hparams['initial_frame_skip']
#)
dataset.setup(task='regression')

train_dataloader = dataset.train_dataloader()
val_dataloader = dataset.val_dataloader()
test_dataloader = dataset.test_dataloader()

model = AQS4(
    joint_features=3,
    joint_count=19,
    joint_expansion=hparams['joint_expansion'],
    layers_count=hparams['layers_count'],
    d_output=1  # Correct-incorrect probability
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
