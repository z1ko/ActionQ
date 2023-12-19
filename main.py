from actionq.dataset.KiMoRe import KiMoReDataset

dataset = KiMoReDataset(
    root_dir='dataset/KiMoRe', 
    exercise=1, 
    subjects=['expert', 'backpain', 'stroke']
)

#dataset.visualize_2d(1)
dataset.visualize_time_series(3)
