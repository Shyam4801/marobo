import pickle
import numpy as np
from numpy.random import default_rng
import pickle
import unittest

from bo.gprInterface import InternalGPR
from bo.utils import Fn, compute_robustness
from bo.sampling import uniform_sampling
from bo.bayesianOptimization.internalBO import InternalBO
from bo import Behavior, PerformBO
from matplotlib import pyplot as plt


class Test_internalBO(unittest.TestCase):
    def test1_internalBO(self):
        def internal_function(X):
            return X[0] ** 2
            # return X[0] ** 2 + X[1] ** 2 + X[2] ** 2

        seed = 12345
        region_support = np.array([[-1, 1]])

        gpr_model = InternalGPR()
        bo = InternalBO()

        opt = PerformBO(
            test_function=internal_function,
            init_budget=50,
            max_budget=70,
            region_support=region_support,
            seed=seed,
            behavior=Behavior.MINIMIZATION,
            init_sampling_type="lhs_sampling"
        )

        data = opt(bo, gpr_model)

        assert np.array(data.history, dtype=object).shape[0] == 70
        assert np.array(data.history, dtype=object).shape[1] == 3
        

    def test2_internalBO(self):
        def internal_function(X):
            return X[0] ** 2 + X[1] ** 2
            # return X[0] ** 2 + X[1] ** 2 + X[2] ** 2

        region_support = np.array([[-1, 1], [-1, 1]])
        seed = 12345
        gpr_model = InternalGPR()
        bo = InternalBO()

        opt = PerformBO(
            test_function=internal_function,
            init_budget=50,
            max_budget=70,
            region_support=region_support,
            seed=seed,
            behavior=Behavior.MINIMIZATION,
            init_sampling_type="lhs_sampling"
        )

        data = opt(bo, gpr_model)

        assert np.array(data.history, dtype=object).shape[0] == 70
        assert np.array(data.history, dtype=object).shape[1] == 3

    def test3_internalBO(self):
        def internal_function(X):
            return X[0] ** 2 + X[1] ** 2 + X[2] ** 2

        
        region_support = np.array([[-1, 1], [-2, 2], [-3, 3]])
        seed = 12345
        gpr_model = InternalGPR()
        bo = InternalBO()

        opt = PerformBO(
            test_function=internal_function,
            init_budget=50,
            max_budget=70,
            region_support=region_support,
            seed=seed,
            behavior=Behavior.MINIMIZATION,
            init_sampling_type="lhs_sampling"
        )

        data = opt(bo, gpr_model)

        assert np.array(data.history, dtype=object).shape[0] == 70
        assert np.array(data.history, dtype=object).shape[1] == 3

    def test4_internalBO(self):
        def internal_function(X):
            return X[0] ** 2
            # return X[0] ** 2 + X[1] ** 2 + X[2] ** 2

        region_support = np.array([[-1, 1]])

        seed = 12345
        gpr_model = InternalGPR()
        bo = InternalBO()

        opt = PerformBO(
            test_function=internal_function,
            init_budget=50,
            max_budget=70,
            region_support=region_support,
            seed=seed,
            behavior=Behavior.MINIMIZATION,
            init_sampling_type="lhs_sampling"
        )

        data = opt(bo, gpr_model)

        assert np.array(data.history, dtype=object).shape[0] == 70
        assert np.array(data.history, dtype=object).shape[1] == 3

    def test5_internalBO(self):
        def internal_function(X):
            return X[0] ** 2 + X[1] ** 2
            # return X[0] ** 2 + X[1] ** 2 + X[2] ** 2

        region_support = np.array([[-1, 1], [-1, 1]])

        seed = 12345
        gpr_model = InternalGPR()
        bo = InternalBO()

        opt = PerformBO(
            test_function=internal_function,
            init_budget=50,
            max_budget=70,
            region_support=region_support,
            seed=seed,
            behavior=Behavior.MINIMIZATION,
            init_sampling_type="lhs_sampling"
        )

        data = opt(bo, gpr_model)

        assert np.array(data.history, dtype=object).shape[0] == 70
        assert np.array(data.history, dtype=object).shape[1] == 3

    def test6_internalBO(self):
        def internal_function(X):
            return X[0] ** 2 + X[1] ** 2 + X[2] ** 2 - 1

        region_support = np.array([[-1, 1], [-2, 2], [-3, 3]])
        

        seed = 12345
        gpr_model = InternalGPR()
        bo = InternalBO()

        opt = PerformBO(
            test_function=internal_function,
            init_budget=50,
            max_budget=70,
            region_support=region_support,
            seed=seed,
            behavior=Behavior.MINIMIZATION,
            init_sampling_type="lhs_sampling"
        )

        data = opt(bo, gpr_model)

        assert np.array(data.history, dtype=object).shape[0] == 70
        assert np.array(data.history, dtype=object).shape[1] == 3

if __name__ == "__main__":
    unittest.main()