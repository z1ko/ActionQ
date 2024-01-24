
import os
import argparse
import lightning as L
from lightning.pytorch.loggers import CometLogger

from actionq.model.s4 import AQS4
from actionq.dataset.KiMoRe import KiMoReDataModule
from actionq.model.regression import ActionQ

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=100, type=int)
parser.add_argument('-lr', '--learning_rate', default=0.001)
args = parser.parse_args()

# Experiment configuration
parameters = {
    # Dimension of a batch
    'batch_size': 8,
    # Number of frames for each sample
    'window_size': 250,
    # How many layers of S4 are used to model the time sequence
    'layers_count': 8,
    # Expansion of each joint features
    'joint_expansion': 16
}

dataset = KiMoReDataModule(
    batch_size=parameters['batch_size'],
    exercise=1,
    subjects=['expert', 'non-expert', 'stroke'],
    window_size=parameters['window_size'],
    features=['pos_x', 'pos_y', 'pos_z'],
    features_expansion=True
)
dataset.setup(task='regression')

train_dataloader = dataset.train_dataloader()
val_dataloader = dataset.val_dataloader()
test_dataloader = dataset.test_dataloader()

# TODO: Move API key to env variable
logger = CometLogger(
    api_key='Zm5C4f6cFAA93SbzYWspFI8hg', 
    save_dir='logs', 
    project_name='AQS4',
    experiment_name=f'{parameters}'
)

model = AQS4(
    joint_features=3, 
    joint_count=19, 
    joint_expansion=parameters['joint_expansion'], 
    layers_count=parameters['layers_count'], 
    d_output=1 # Clinical Total Score
)

model = ActionQ(model, lr=args.learning_rate, maximum_score=50.0, weight_decay=0.001)
trainer = L.Trainer(max_epochs=args.epochs, logger=logger)
trainer.fit(model, 
    train_dataloaders=train_dataloader, 
    val_dataloaders=val_dataloader,
)

trainer.test(
    dataloaders=train_dataloader
)