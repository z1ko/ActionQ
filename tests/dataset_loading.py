import pandas as pd
import numpy as np
import torch

EXERCISE = 1
FEATURES = ['pos_x', 'pos_y', 'pos_z']

samples = []

# Load samples from processed dataset
samples_path = 'data/processed/kimore_samples.parquet.gzip'
df = pd.read_parquet(samples_path)

# Keeps only data of exercise 1
samples_df = df[df['exercise'] == EXERCISE]

# Create samples for training
grouped = samples_df.groupby(['type', 'subject'])
for _, group in grouped:

    # Keeps only features and index by frame and joint id
    group.set_index(['frame', 'joint'], inplace=True)
    group = group[FEATURES]

    # Get dimensionality
    frames_count = len(group.index.get_level_values(0).unique())
    joints_count = len(group.index.get_level_values(1).unique())

    try:
        result = np.reshape(group, (frames_count, joints_count, 3))
        samples.append(result)
    except:
        print("ERROR CONVERTING!")
        print(group)
        input()

print(f'samples: {len(samples)}')
for sample in samples:
    print(f'\tsize: {sample.shape}')
