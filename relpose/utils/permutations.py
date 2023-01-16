def get_permutations(num_images):
    for i in range(num_images):
        for j in range(num_images):
            if i != j:
                yield (i, j)
