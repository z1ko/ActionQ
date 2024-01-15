import pandas as pd
import numpy as np

EXERCISE = 1
FEATURES = ['pos_x', 'pos_y', 'pos_z']

# Load samples from processed dataset
samples_path = 'data/processed/kimore_samples.parquet.gzip'
df = pd.read_parquet(samples_path)

# Keeps only data of exercise 1
samples_df = df[df['exercise'] == EXERCISE]
samples_df = samples_df.drop(columns=['exercise'])

# Create samples for training
grouped = samples_df.groupby(['type', 'subject'])
for _, group in grouped:
    sample = group.drop(columns=['type', 'subject'])
    sample['features'] = sample[FEATURES].values.tolist()
    sample = sample.drop(columns=FEATURES).reset_index(drop=True)
    print(sample)

