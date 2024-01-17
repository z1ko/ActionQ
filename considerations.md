# Considerations

- For now skeleton temporal data is rescaled to the range [-1, 1], considering the maximum and minimum extensions of all body joints (max and min over all joints coordinates of a sample). This is useful to allow an easier comparison between skeletons of different samples, but may be biased.

- A low-pass filter may be applied to the joint coords time series to reduce noise

- Only 2d position coords are used, maybe 3d coords or bone angles are better features. Angles are probably easier to normalize across all samples compared to spatial positions.

- Maybe we could generate new training samples by applying a small rotation on the y axis (if we have good enough quality depth data) and a small zx plane traslation. If using S4 we could also slow down or speedup the action.

# Training

- We have to few samples, the network should be small or it will hardly train
