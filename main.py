import argparse
import lightning as L
from lightning.pytorch.loggers import CometLogger

from actionq.model.s4 import AQS4
from actionq.dataset.KiMoRe import KiMoReDataModule
from actionq.model.classification import ActionClassifier

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=100, type=int)
parser.add_argument('-lr', '--learning_rate', default=0.001)
args = parser.parse_args()

# Experiment configuration
parameters = {
    # Dimension of a batch
    'batch_size': 4,
    # Number of frames for each sample
    'window_size': 400,
    # How many layers of S4 are used to model the time sequence
    'layers_count': 4,
    # Expansion of each joint features
    'joint_expansion': 16
}

data = KiMoReDataModule(
    batch_size=parameters['batch_size'],
    features=['pos_x', 'pos_y'],
    window_size=parameters['window_size']
)
data.setup(task='classification')

# TODO: Move API key to env variable
logger = CometLogger(
    api_key='Zm5C4f6cFAA93SbzYWspFI8hg', 
    save_dir='logs', 
    project_name='AQS4-Classifier'
)

model = AQS4(
    joint_features=2, 
    joint_count=19, 
    joint_expansion=parameters['joint_expansion'], 
    layers_count=parameters['layers_count'], 
    d_output=5 # Classes
)

module = ActionClassifier(model, lr=args.learning_rate, weight_decay=0.01)
trainer = L.Trainer(max_epochs=args.epochs, logger=logger)
trainer.fit(module, 
    train_dataloaders=data.train_dataloader(), 
    val_dataloaders=data.val_dataloader(),
)