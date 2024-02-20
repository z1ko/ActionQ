import torch
import lightning
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pprint
import einops as ein
from time import perf_counter

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
    joint_expansion=256,
    temporal_layers_count=1,
    spatial_layers_count=0,
    output_dim=1,
    # skeleton=skeleton_adj_matrix(),
    dropout=0.1,
    r_min=0.4,
    r_max=0.8,
    mode='predict'
)

# Load checkpoint
model = ActionQ.load_from_checkpoint(checkpoint_path=args.checkpoint, model=model)
model.eval().to(torch.device('cuda'))

# Check entire dataset
datamodule = KiMoReDataModule(
    exercise=1,
    subjects=['expert', 'non-expert', 'stroke', 'backpain', 'parkinson'],
    features=['pos_x', 'pos_y', 'pos_z'],
    window_size=600,
    window_delta=50,
    features_expansion=True,
    normalize=False,
    batch_size=1
)

device = torch.device('cuda')

datamodule.setup('regression')
data = datamodule.val_dataloader()

def evaluate_frames(frames, state):
    deltas = np.zeros(frames.shape[0])
    results = np.zeros(frames.shape[0])

    for i in range(frames.shape[0]):
        frame = frames[i].to(device)
        beg_t = perf_counter()
        y = model.predict_step(frame, state)
        deltas[i] = perf_counter() - beg_t
        results[i] = y.item()

    return results, deltas

frames0, target0 = list(data)[0]
frames1, target1 = list(data)[10]


frames = torch.cat((frames0, frames1), dim=1)
frames = frames.squeeze(0)

# State of the system
state = torch.complex(torch.zeros(256), torch.zeros(256))
state = ein.repeat(state, 'c -> t j c', t=1, j=19)
state = state.to(device)

results, deltas = evaluate_frames(frames, state)

delta = deltas.mean()
fps = 1.0 / delta
print(f'prediction time (ms): {delta}, fps: {fps}')
print(f'targets = {(target0, target1)}')


plt.plot(results)
plt.ylim(0.0, 50.0)
plt.show()

# trainer = lightning.Trainer()
# predictions = trainer.predict(model, data)
# predictions = list(map(lambda item: [item[0].item(), item[1].item()], predictions))
# predictions = np.array(predictions)
# plt.plot(predictions[:, 0], label='predict')
# plt.plot(predictions[:, 1], label='true')
# plt.legend()
# plt.show()
