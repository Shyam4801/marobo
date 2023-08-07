import enum
import time
import numpy as np
from attr import frozen
from typing import Callable, Any
from numpy.typing import NDArray
from .utils.logged import *

from .bayesianOptimization import BOSampling
from .utils import Fn, compute_robustness
from .sampling import uniform_sampling, lhs_sampling
from .behavior import Behavior

# logger = MyLogger("expLog.log").get_logger()
logging.basicConfig()

@frozen(slots=True)
class BOResult:
    """Data class that represents the result of a uniform random optimization.

    Attributes:
        average_cost: The average cost of all the samples selected.
    """
    history: Any
    agent_history: Any
    optimization_time: float


class PerformBO:
    def __init__(
            self, 
            test_function: Callable,
            init_budget: int,
            max_budget: int,
            region_support: NDArray,
            seed,
            num_agents: int,
            behavior:Behavior = Behavior.MINIMIZATION,
            init_sampling_type = "lhs_sampling",
            logger = None
        ):
            """Internal BO Model

            Args:
                test_function: Function of System Under Test.
                init_budget: Number of samples in Initial Budget,
                max_budget: Maximum budget
                region_support: Min and Max of all dimensions
                seed: Set seed for replicating Experiment
                behavior: Set behavior to Behavior.MINIMIZATION or Behavior.FALSIFICATION
                init_sampling_type: Choose between "lhs_sampling" or "uniform_sampling"

            Returns:
                x_complete
                y_complete
                x_new
                y_new
            """
            
            if max_budget < init_budget:
                raise ValueError("Max Budget cannot be less than Initial Budget")

            self.tf_wrapper = Fn(test_function)
            self.init_budget = init_budget
            self.max_budget = max_budget
            self.region_support = region_support
            self.seed = seed
            self.rng = np.random.default_rng(seed)
            self.init_sampling_type = init_sampling_type
            self.behavior = behavior
            self.logger = logger
            self.num_agents = num_agents
            
    # @log(my_logger=logger)
    def __call__(self, bo_model, gpr_model):
        start_time = time.perf_counter()
        tf_dim = self.region_support.shape[0]
        # self.logger.info("Bounds : {} ".format(self.region_support),"Dimensions: {} ".format(tf_dim))
        bo_routine = BOSampling(bo_model)
        

        if self.init_sampling_type == "lhs_sampling":
            x_train = lhs_sampling(self.init_budget, self.region_support, tf_dim, self.rng)
        elif self.init_sampling_type == "uniform_sampling":
            x_train = uniform_sampling(self.init_budget, self.region_support, tf_dim, self.rng)
        else:
            raise ValueError(f"{self.init_sampling_type} not defined. Currently only Latin Hypercube Sampling and Uniform Sampling is supported.")
        
        y_train, falsified = compute_robustness(x_train, self.tf_wrapper, self.behavior)

        if not falsified:
            print("No falsification in Initial Samples. Performing BO now")
            falsified, rg, plot_res = bo_routine.sample(self.tf_wrapper, self.max_budget - self.init_budget, x_train, y_train, self.region_support, gpr_model, self.rng, self.num_agents)

        return BOResult(self.tf_wrapper.point_history, self.tf_wrapper.agent_point_history,time.perf_counter()-start_time), rg, plot_res