import numpy as np
from bo import Behavior, PerformBO
from bo.bayesianOptimization import InternalBO
from bo.gprInterface import InternalGPR
from bo.interface import BOResult


# def internal_function(X):
#             return X[0] ** 2 + X[1] ** 2 -1

# init_reg_sup = np.array([[-1, 1], [-2, 2]])                          # cartesian prod is -1,-2 ; -1,2 ; 1,-2 ; 1,2
# def internal_function(X):
#     return (X[0] - 2)**2 + (X[1] - 2)**2   #glob min (2,2) Local minimum at (0.5, 0.5)

# init_reg_sup = np.array([[-5, 5], [-5, 5]])   

# def internal_function(X): #Branin with unique glob min -  9.42, 2.475
#         # if X.shape[1] != 2:
#         #     raise Exception('Dimension must be 2')
#         # d = 2
#         # if lb is None or ub is None:
#         #     lb = np.full((d,), 0)
#         #     ub = np.full((d,), 0)
#         #     lb[0] = -5
#         #     lb[1] = 0
#         #     ub[0] = 10
#         #     ub[1] = 15
#         # x = from_unit_box(x, lb, ub)
#         x1 = X[0]
#         x2 = X[1]
#         t = 1 / (8 * np.pi)
#         s = 10
#         r = 6
#         c = 5 / np.pi
#         b = 5.1 / (4 * np.pi ** 2)
#         a = 1
#         term1 = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
#         term2 = s * (1 - t) * np.cos(x1)
#         l1 = 5 * np.exp(-5 * ((x1+3.14)**2 + (x2-12.27)**2))
#         l2 = 5 * np.exp(-5 * ((x1+3.14)**2 + (x2-2.275)**2))
#         return term1 + term2 + s + l1 + l2

# init_reg_sup = np.array([[-5, 10], [-5, 15]])

# Evaluated by default from [-4, 5]^d
def from_unit_box(x, lb, ub):
    return lb + (ub - lb) * x

def internal_function(x, lb=None, ub=None): #ackley
    n = len(x)
    sum_sq_term = -0.2 * np.sqrt((1/n) * np.sum(x**2))
    cos_term = np.sum(np.cos(2 * np.pi * x))
    return -20 * np.exp(sum_sq_term) - np.exp(cos_term) + 20 + np.exp(1)

range_array = np.array([[-2, 2]])  # Range [-4, 5] as a 1x2 array
init_reg_sup = np.tile(range_array, (10, 1))  # Replicate the range 10 times along axis 0

# def internal_function(X):
#     return X[0]**4 + X[1]**4 - 4*X[0]*X[1] + 1

# init_reg_sup = np.array([[-5, 5], [-5, 5]]) 



optimizer = PerformBO(
    test_function=internal_function,
    init_budget=10,
    max_budget=80,
    region_support=init_reg_sup,
    seed=12345,
    behavior=Behavior.MINIMIZATION,
    init_sampling_type="lhs_sampling"
)

z = optimizer(bo_model=InternalBO(), gpr_model=InternalGPR())
history = z.history
time = z.optimization_time

print(np.array(history, dtype=object))
print(f"Time taken to finish iterations: {round(time, 3)}s.")