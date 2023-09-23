import numpy as np
import numpy.typing as npt
from ..utils.volume import compute_volume

def uniform_sampling(
    num_samples: int, region_support: npt.NDArray, tf_dim: int, rng
) -> np.array:
    """Sample *num_samples* points within the *region_support* which has a dimension as mentioned below.

    Args:
        num_samples: Number of points to sample within the region bounds.
        region_support: The bounds of the region within which the sampling is to be done.
                                    Region Bounds is N x O where;
                                        N = tf_dim (Dimensionality of the test function);
                                        O = Lower and Upper bound. Should be of length 2;
        tf_dim: The dimensionality of the region. (Dimensionality of the test function)

    Returns:
        np.array: 3d array with samples between the bounds.
                    Size of the array will be N x O
                        N = num_samples
                        O = tf_dim (Dimensionality of the test function)
    """

    if region_support.shape[0] != tf_dim:
        raise ValueError(f"Region Support has wrong dimensions. Expected {tf_dim}, received {region_support.shape[0]}")
    if region_support.shape[1] != 2:
        raise ValueError("Region Support matrix must be MxNx2")
    
    if not np.alltrue(region_support[:,1]-region_support[:,0] >= 0):
        raise ValueError("Region Support Z-pairs must be in increasing order")

    raw_samples = np.apply_along_axis(
        lambda bounds: rng.uniform(bounds[0], bounds[1], num_samples),
        1,
        region_support,
    )

    samples = []
    for sample in raw_samples:
        samples.append(sample)

    return np.array(samples).T


def sample_from_discontinuous_region(num_samples, regions, region_support, tf_dim, rng, volume=True ):
        filtered_samples = np.empty((0,tf_dim))
        # threshold = 0.3
        total_volume = compute_volume(region_support)
        vol_dic = {}
        for reg in regions:
            # print('inside vol dict ', reg.input_space)
            v =  compute_volume(reg.input_space) / total_volume
            if v != np.inf:
                vol_dic[reg] = v
            else:
                vol_dic[reg] = 1

        vol_dic_items = sorted(vol_dic.items(), key=lambda x:x[1])
        # print('vol dict :',vol_dic_items, total_volume)
        for v in vol_dic_items:
            tsmp = uniform_sampling(int(num_samples*v[1]), v[0].input_space, tf_dim, rng)
            
            filtered_samples = np.vstack((filtered_samples,tsmp))

        # print('filtered_samples: ,', filtered_samples)
        return filtered_samples
