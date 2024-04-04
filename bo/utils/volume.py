
import numpy as np 

def compute_volume(region):
    dimensions = len(region)
    volume = 1
    # print("dimensions: ",dimensions)
    for i in range(dimensions):
        # print('volume: ',volume)
        volume *= np.float64(region[i][1] - region[i][0])
        
    return volume

def compute_volume_per_dim(region):
    dimensions = len(region)
    vperDim = []
    volume = 1
    # print("dimensions: ",dimensions)
    for i in range(dimensions):
        # print('volume: ',volume)
        volume = np.float64(region[i][1] - region[i][0])
        vperDim.append(volume)
    idxvperDim = np.argsort(vperDim)
    idxvperDim = np.flip(idxvperDim)
    return idxvperDim