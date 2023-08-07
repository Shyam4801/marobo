

def compute_volume(region):
    dimensions = len(region)
    volume = 1
    for i in range(dimensions):
        volume *= region[i][1] - region[i][0]
    return volume

