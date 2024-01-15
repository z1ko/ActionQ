
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import sys

# Name of all skeleton joints
SKELETON_JOINT_NAMES = [
    'spine_base',
    'spine_mid',
    'neck',
    'head',
    'shoulder_left',
    'elbow_left',
    'wrist_left',
    'hand_left',
    'shoulder_right',
    'elbow_right',
    'wrist_right',
    'hand_right',
    'hip_left',
    'knee_left',
    'ankle_left',
    'hip_right',
    'knee_right',
    'ankle_right',
    'spine_shoulder',
    'TODO(?)'
]

if len(sys.argv) != 3:
    print('args: <subject> <exercise>')
    exit(1)

arg_subject = sys.argv[1]
arg_exercise = int(sys.argv[2])

# ===================================================================================
# Style configuration

mpl.style.use('seaborn-v0_8-deep')

mpl.rcParams['figure.figsize'] = (20, 5)
mpl.rcParams['figure.dpi'] = 100

# ===================================================================================
# Load dataset

samples_df = pd.read_parquet('data/processed/kimore_samples.parquet.gzip')

# ===================================================================================
# Plot single subject, single exercise joint's features evolution


def _plot_joint(df, subject, exercise, joint, ax):
    joints_df = df.query(f"subject == '{subject}'").query(f"exercise == {exercise}").query(f"joint == {joint}").reset_index()
    joints_df[['pos_x', 'pos_y', 'pos_z']].plot(ax=ax)
    ax.set_title(f'joint = {SKELETON_JOINT_NAMES[joint]}')
    ax.set_ylabel('value')
    ax.set_xlabel('frames')
    ax.legend()


joints = list(samples_df['joint'].unique())
fig, ax = plt.subplots(nrows=(len(joints) + 4 - 1) // 4, ncols=4, sharex=True, figsize=(20, 10))
fig.suptitle(f'subject = {arg_subject}, exercise = {arg_exercise}', fontsize=16)
fax = ax.flatten()

for i, joint in enumerate(joints):
    _plot_joint(samples_df, arg_subject, arg_exercise, joint, fax[i])

fig.tight_layout()
fig.subplots_adjust(top=0.88)
plt.show()
