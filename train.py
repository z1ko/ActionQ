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
    'batch_size': 16,
    # Number of frames for each sample
    'window_size': 250,
    # How many layers of S4 are used to model the time sequence
    'layers_count': 4,
    # Expansion of each joint features
    'joint_expansion': 4
}

dataset = KiMoReDataModule(
    batch_size=parameters['batch_size'],
    exercise=1,
    subjects=None,
    window_size=parameters['window_size'],
    features=['pos_x', 'pos_y'],
)
dataset.setup()

train_dataloader = dataset.train_dataloader()
val_dataloader = dataset.val_dataloader()
test_dataloader = dataset.test_dataloader()

#logger = NeptuneLogger(
#    api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzYmJmOTJiMy01N2NmLTRmNGYtYTE5NC1iZjliODNjZjBlNTUifQ==',
#    project='PARCO/ActionQ'
#)

# TODO: Move API key to env variable
logger = CometLogger(
    api_key='Zm5C4f6cFAA93SbzYWspFI8hg', 
    save_dir='logs', 
    project_name='AQS4',
    experiment_name=f'{parameters}'
)

model = AQS4(
    joint_features=2, 
    joint_count=19, 
    joint_expansion=parameters['joint_expansion'], 
    layers_count=parameters['layers_count'], 
    d_output=1 # Clinical Total Score
)

model = ActionQ(model, lr=args.learning_rate, weight_decay=0.01)
trainer = L.Trainer(max_epochs=args.epochs, logger=logger)
trainer.fit(model, 
    train_dataloaders=train_dataloader, 
    val_dataloaders=val_dataloader,
)

trainer.test(
    dataloaders=train_dataloader
)