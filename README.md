What I've learned is that the distribution of the first delta is very very different from the distribution
of the rest of the deltas. Therefore, the model trains to follow the initial distribution, but
ignores the first point. When inference time comes, the model fails the first point and goes and out
distribution and out of bounds, resulting in a batch reconstruction
