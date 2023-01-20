import numpy as np


def get_permutations(num_images):
    permutations = []
    for i in range(num_images):
        for j in range(num_images):
            if i != j:
                permutations.append((i, j))
    return np.array(permutations)
