# =================================================
# This file visualizes distributions of
# the datasets.
# =================================================

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--metrics', action='store_true')
parser.add_argument('-c', '--counts', action='store_true')
parser.add_argument('-k', '--kde', action='store_true')
args = parser.parse_args()

# ===================================================================================
# Style configuration

mpl.style.use('seaborn')
mpl.rcParams['figure.figsize'] = (20, 10)
mpl.rcParams['figure.dpi'] = 100

# ===================================================================================

df = pd.read_parquet('data/processed/kimore_targets.parquet.gzip')
print(df)

# ===================================================================================
# Plot distribution of metrics based on type of patient using boxplots

if args.metrics:

    fig, axs = plt.subplots(nrows=3, sharex=True)
    fig.suptitle('Metrics by type of patient', fontsize=16)
    
    for ax, metric in zip(axs, ['TS', 'PO', 'CF']):
        df.boxplot(column=metric, by=['type'], ax=ax)
        ax.set_xlabel('')
    
    plt.tight_layout()
    plt.show()

# ===================================================================================
# Plot distribution of metrics based on type of patient using KDE

if args.kde:
    
    fig, axs = plt.subplots(nrows=3, sharex=False)
    fig.suptitle('Metrics by type of patient', fontsize=16)

    for ax, metric in zip(axs, ['TS', 'PO', 'CF']):
        df.groupby('type')[metric].plot(kind='kde', ax=ax)
        ax.set_title(metric)
        ax.legend()

    plt.tight_layout()
    plt.show()

# ===================================================================================
# Plot types' samples count

if args.counts:

    df.groupby('type').size().plot(kind='bar')
    plt.suptitle('Number of samples for each type of patient', fontsize=16)
    plt.ylabel('count')
    plt.xticks(rotation=45)
    plt.show()
