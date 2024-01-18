import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import pandas as pd
import numpy as np
import torch
import sys

# Hack assurdo per caricare moduli da altri cartelle
sys.path.append('.')
from actionq.dataset.KiMoRe import rescale_sample

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--velocity', action='store_true')
args = parser.parse_args()

def insert_velocity(df):
    df.set_index(['joint', 'frame'], inplace=True)

    df['vel_x'] = df['pos_x'].diff()
    df['vel_y'] = df['pos_y'].diff()

    # Normalize velocity to [-1, 1]
    magnitude = np.sqrt(df['vel_x']**2 + df['vel_y']**2)
    df['vel_x'] /= magnitude
    df['vel_y'] /= magnitude

    df['vel_m'] = magnitude


df = pd.read_parquet('data/processed/kimore_samples.parquet.gzip')
print(f'successfully loaded dataset')

# Increment features with velocity
if args.velocity:
    insert_velocity(df)

print(df)