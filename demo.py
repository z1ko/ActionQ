import torch
import lightning
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pprint

from actionq.model.lru_model import LRUModel
from actionq.model.regression import ActionQ
from actionq.dataset.KiMoRe import KiMoReDataModule

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=str)

args = parser.parse_args()
pprint.pprint(vars(args))

# TODO: Find out why why need to instantiate this...
model = LRUModel(
    joint_features=6,
    joint_count=19,
    joint_expansion=48,
    temporal_layers_count=8,
    spatial_layers_count=0,
    output_dim=1,
    # skeleton=skeleton_adj_matrix(),
    dropout=0.1,
    r_min=0.4,
    r_max=0.8
)

# Load checkpoint
model = ActionQ.load_from_checkpoint(checkpoint_path=args.checkpoint, model=model)
model.eval().to(torch.device('cuda'))

# Check entire dataset
datamodule = KiMoReDataModule(
    exercise=1,
    subjects=['expert', 'non-expert', 'stroke', 'backpain', 'parkinson'],
    features=['pos_x', 'pos_y', 'pos_z'],
    window_size=200,
    window_delta=50,
    features_expansion=True,
    normalize=False,
    batch_size=1
)

datamodule.setup('regression')
data = datamodule.val_dataloader()
trainer = lightning.Trainer()

predictions = trainer.predict(model, data)
predictions = list(map(lambda item: [item[0].item(), item[1].item()], predictions))
predictions = np.array(predictions)

plt.plot(predictions[:, 0], label='predict')
plt.plot(predictions[:, 1], label='true')

plt.legend()
plt.show()
