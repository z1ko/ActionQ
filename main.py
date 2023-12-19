from actionq.dataset.KiMoRe import KiMoReDataModule, KiMoReDataVisualizer

dataset = KiMoReDataModule(
    root_dir='dataset/KiMoRe',
    batch_size=1,
    subjects=['expert', 'backpain', 'stroke'],
    exercise=1
)

dataset.setup('')
dataloader = dataset.train_dataloader()

visualizer = KiMoReDataVisualizer()
sample = next(iter(dataloader))[0]
visualizer.visualize_2d(sample)

#dataset.visualize_2d(1)
#dataset.visualize_time_series(3)
