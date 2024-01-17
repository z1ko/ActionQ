from lightning.pytorch.loggers import CometLogger, TensorBoardLogger
import lightning as L
from actionq.model.s4 import AQS4
import matplotlib.pyplot as plt
import torch.functional as F
from torch.utils.data import DataLoader
from actionq.dataset.KiMoRe import KiMoReDataModule, KiMoReDataVisualizer, KiMoReDataset
from einops import rearrange
from actionq.model.regression import ActionQ

#D = KiMoReDataset(exercise=1, window_size=200, rescale_samples=True)
#print(f'dataset size: {len(D)}')
#
#visualizer = KiMoReDataVisualizer()
#for i, sample in enumerate(D):
#    visualizer.visualize_2d(sample)
#    if i > 5: break
#
#exit(0)

dataset = KiMoReDataModule(
    batch_size=2,
    exercise=1,
    window_size=400
)

dataset.setup()

#
# visualizer = KiMoReDataVisualizer()
# for batch in dataloader:
#    sample = batch[0]
#    visualizer.visualize_2d(sample)
#

train_dataloader = dataset.train_dataloader()
val_dataloader = dataset.val_dataloader()
test_dataloader = dataset.test_dataloader()

hyperparameters = {
    # How many layers of S4 are used to model the time sequence
    'layers_count': 6,
    # Expansion of each joint features
    'joint_expansion': 32
}

# TODO: Move API key to env variable
logger = CometLogger(
    api_key='Zm5C4f6cFAA93SbzYWspFI8hg', 
    save_dir='logs', 
    project_name='AQS4',
    experiment_name=f'{hyperparameters}'
)

model = AQS4(
    joint_features=3, 
    joint_count=19, 
    joint_expansion=hyperparameters['joint_expansion'], 
    layers_count=hyperparameters['layers_count'], 
    d_output=3
)

model = ActionQ(model, lr=0.001, weight_decay=0.01)
trainer = L.Trainer(max_epochs=50, logger=logger, log_every_n_steps=5)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
