
from .rolloutEI import simulate
import yaml
from joblib import Parallel, delayed
from ..utils.savestuff import *

with open('config.yml', 'r') as file:
    configs = yaml.safe_load(file)


class RolloutRoutine:
    def __init__(self) -> None:
        pass


    def run(self, m, xroots, globalGP, num_agents, tf, tf_dim, behavior, rng):
        """
        Parallelize configuration rollout

        Args: 
            m: ith agent to rollout fixing (1 to i-1) agent configurations
            xroots: Configurations to rollout
            num_agents: Number of agents
            test_function_dimension: The dimensionality of the region. (Dimensionality of the test function)

        Return:
            Xs_roots: Simulated Agent Configuration 
            F_nc: Approximate Q-factor

        """
        # Number of times to simulate the Rollout (N^{RO})
        mc = configs['sampling']['mc_iters']
        if configs['parallel']:
            res = Parallel(n_jobs=-1)(delayed(simulate)(m, Xs_root_item, globalGP=globalGP, mc_iters=mct, num_agents=num_agents, tf=tf, tf_dim=tf_dim, behavior=behavior, horizon=4, rng=rng) for mct in range(1,mc) for (Xs_root_item) in xroots )
            Xs_roots = [res[i][0] for i in range(len(res))]
            F_nc = [res[i][1] for i in range(len(res))]

        else:
            Xs_roots=[]
            F_nc = []
            for xr in xroots:
                res, f_nc = simulate(m, xr, globalGP=globalGP, mc_iters=2, num_agents=num_agents, tf=tf, tf_dim=tf_dim, behavior=behavior, horizon=4, rng=rng)
                Xs_roots.append(res)
                F_nc.append(f_nc)
        
        return Xs_roots, F_nc