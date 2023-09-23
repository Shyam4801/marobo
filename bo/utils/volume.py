
import numpy as np 

def compute_volume(region):
    dimensions = len(region)
    volume = 1
    # print("dimensions: ",dimensions)
    for i in range(dimensions):
        # print('volume: ',volume)
        volume *= np.float128(region[i][1] - region[i][0])
    return volume

