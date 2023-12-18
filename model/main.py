from dataset.KiMoRe import KiMoReDataset

dataset = KiMoReDataset(
    root_dir='dataset/KiMoRe', 
    exercise=1, 
    subjects=['expert']
)

dataset.visualize_2d(1)
#dataset.visualize_time_series(0)
