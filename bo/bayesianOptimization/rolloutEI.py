import numpy as np
from tqdm import tqdm
from copy import deepcopy
import time 

from .internalBO import InternalBO
from ..agent.treeOperations import * 
from ..sampling import uniform_sampling, sample_from_discontinuous_region, lhs_sampling
import yaml
from ..utils.savestuff import *


with open('config.yml', 'r') as file:
    configs = yaml.safe_load(file)
  
class RolloutEI(InternalBO):
    def __init__(self) -> None:
        pass
           
    # Rollout from the current state of the agents
    def rollout(self, m, root, globalGP, horizon, num_agents, tf, tf_dim, behavior, rng):

        """Simulated agent outcomes (Step 1: Configuration generation followed by Step 2: Configuration rollout)

        Args: 
            m: ith agent to rollout fixing (1 to i-1) agent configurations
            root: Configuration to rollout 
            horizon: Steps to simulate the agent outcomes
            num_agents: Number of agents
            test_function_dimension: The dimensionality of the region. (Dimensionality of the test function)

        Return:
            f_ro: Approximate Q-factor of the configuration after simulating for 'h' horizons
                    
        """
        self.num_agents = num_agents
        self.tf = tf
        self.tf_dim = tf_dim
        self.behavior = behavior
        reward = 0
        actm = m 
        h = horizon
        self.tf_dim = len(root.input_space)
        rl_root = root 
        
        
        # Simulate the state of the agents by keeping a copy of the locations sampled
        tmpGP = deepcopy(globalGP)
        ytr = np.min(tmpGP.dataset.y_train)
        # Rollout the agent movements for 'h' horizons
        while(True):
            totalVolume = 0
            xt = rl_root.find_leaves() 
            for reg in xt:
                totalVolume += reg.getVolume()

            agents = []
            for r in xt:
                if r.getStatus() == RegionState.ACTIVE.value:
                    agents.append(r)
            
            agents = sorted(agents, key=lambda x: x.agentId)

            
            cfgroots = [rl_root]
            # f_ro, rl_root = self.forward(actm, tmpGP, xt, cfgroots, h, totalVolume, rng)
            # # Get the different possible splits for the chosen config rl_root
            # roots = getRootConfigs(m, rl_root, tmpGP, 1, self.num_agents, self.tf_dim, tf, behavior, rng)
            # Perform rollout one agent at a time
            for m in range(actm, self.num_agents):
                # Get the config with the min approx Q factor
                f_ro, rl_root = self.forward(m, tmpGP, xt, cfgroots, h, totalVolume, rng)
                print('m seq of minz :', m, f_ro, 'actm : ',actm)                  
                # Get the different possible splits for the chosen config rl_root
                roots = getRootConfigs(m, rl_root, tmpGP, 1, self.num_agents, self.tf_dim, tf, behavior, rng)  
                # print('% tree ')
                # print_tree(rl_root)
                # print('-'*20)
                # if m == self.num_agents:
                #     break

                
                # print(f'tree after  getRootConfigs {m}')
                # print_tree(roots[0])
                # print('-'*20)
                # Create different possbile agent assignments and next set of samples to evaluate
                xroots, agentModels, tmpGP = genSamplesForConfigsinParallel(m, tmpGP, configs['configs']['smp'], self.num_agents, roots, "lhs_sampling", self.tf_dim, self.tf, self.behavior, rng)
                # genSamplesForConfigs(m, globalGP, self.num_agents, roots, "lhs_sampling",self.tf_dim, self.tf, self.behavior, rng)
                cfgroots  = np.hstack((xroots))
                agentModels  = np.hstack((agentModels))

                # Check if the observations and samples for each configuration are not empty
                # Build the local GPs for the respective configs based on locations sampled during simulation
                for crt in cfgroots:
                    for id, l in enumerate(crt.find_leaves()): 
                        localGP = Prior(tmpGP.dataset, l.input_space)
                        try:
                            assert localGP.checkPoints(tmpGP.dataset.x_train[l.obsIndices]) == True
                        except AssertionError:
                            print(l.__dict__)
                            exit(1)


                    for a in crt.find_leaves():    
                        if a.getStatus() == RegionState.ACTIVE.value:
                            if len(a.obsIndices) == 0:
                                parent = find_parent(crt, a)
                                a.model = deepcopy(parent.model)
                                
                                actregSamples = lhs_sampling(self.tf_dim*2 , a.input_space, self.tf_dim, rng)  #self.tf_dim*10
                                mu, std = self._surrogate(a.model, actregSamples)  #agent.simModel
                                actY = []
                                for i in range(len(actregSamples)):
                                    f_xt = np.random.normal(mu[i],std[i],1)
                                    actY.append(f_xt)
                                actY = np.hstack((actY))
                                
                                tmpGP.dataset = tmpGP.dataset.appendSamples(actregSamples, actY)
                            # else:
                            localGP = Prior(tmpGP.dataset, a.input_space)
                            a.model , a.obsIndices = localGP.buildModel() #updateObs()   

                 
            
            print(f'* end of {m} minz'*10)
            # print('% tree ')
            # print_tree(roots[0])
            # print('-'*20)
            print('tmp gp end of horizon shape ',tmpGP.dataset.x_train.shape)

            # print()
            # print(f'- end of {h}'*20)
            # print_tree(rl_root)
            # print()
            h -= 1
            if h <= 0 :
                break
                # exit(1)
        fincfgIdx = np.random.randint(len(roots))
        # print('% tree ')
        # print_tree(roots[fincfgIdx])
        # print('-'*20)
        return roots[fincfgIdx], f_ro
                
    # Function to get the config with min approx Q factor across the active regions
    def forward(self, m, tmpGP, xt, roots, h, totalVolume, rng):
        """Configuration evaluation 

        Args: 
            m: ith agent to rollout fixing (1 to i-1) agent configurations
            tmpGP: Copy of the observations encountered so far 
            h: Steps to simulate the agent outcomes
            xt: Partitioned regions (Active and inactive leaf nodes)

        Return:
            f_ro: Max EI among the agents
            f_g: Configurtion correcponding to the max EI
                    
        """
        f_ro = np.float64('inf')
        f_g = None
        for ix, rl_root in enumerate(roots):
            f_cfg, mincfg = self.get_cfg_EI(m, tmpGP, xt, rl_root, h, totalVolume, rng)
            if f_ro > f_cfg:
                f_ro = f_cfg
                f_g = mincfg

        return f_ro, f_g
    
    # Function to evaluate the samples for each config  
    def get_cfg_EI(self, m, tmpGP, xt, rl_root, h, totalVolume, rng):
        xt = rl_root.find_leaves()
        xt = sorted(xt, key=lambda x: (x.getStatus() == RegionState.INACTIVE.value, x.agentId if x.getStatus() == RegionState.ACTIVE.value else float('inf')))
        agents = []
        for r in xt:
            if r.getStatus() == RegionState.ACTIVE.value:
                agents.append(r)
        
        agents = sorted(agents, key=lambda x: x.agentId)

        # Get the f* among all the active regions 
        ytr = self.get_min_across_regions(agents, tmpGP) 
        
        for a in xt[m:self.num_agents]:
            # a = reg
            ix = a.agentId
            # Local GP of agent 
            model = a.model
            if model == None:
                print(' No model !')
                print('h: ', h)

                exit(1)

            for reg in xt[m:]:
                # evaluate the samples in the active region
                if reg.status == RegionState.ACTIVE.value:
                    # An extension to use the common parent GP instead of local GP
                    if a != reg:
                        commonReg = find_common_parent(rl_root, reg, a)
                        model = commonReg.model

                    if model == None:
                        print(' No common model b/w !', a.simReg.input_space, reg.input_space)
                        print('h: ', h)
                        exit(1)

                    # Calculate the cost using EI to facilitate jump
                    xtr = reg.samples.x_train[reg.smpIndices]
                    smpEIs = (-1 * self.cost(xtr, ytr, model, "multiple"))
                    maxEI = np.array([xtr[np.argmin(smpEIs), :]])
                    # Add the location with min cost to the local GPs
                    if a == reg:
                        # Sampling decision: Appending the sample to the observation set to update the local GPs
                        ytr = tmpGP.dataset.y_train[reg.obsIndices] # region f*
                        localSmpEIs = (-1 * self.cost(xtr, ytr, model, "multiple"))
                        maxEI = np.array([xtr[np.argmin(localSmpEIs), :]])
                        mu, std = self._surrogate(model, maxEI)
                        f_xt = np.random.normal(mu,std,1)
                        tmpGP.dataset.appendSamples(maxEI, f_xt)

                        # Keeping track of the sample predicted function values to calculate config cost | Might feel unnecessary but just a way to increase isolation
                        mu, std = self._surrogate(model, xtr)
                        for i in range(len(xtr)):
                            f_xt = np.random.normal(mu[i],std[i],1)
                            reg.yOfsmpIndices[i] = f_xt
                    
                    reg.rewardDist[ix] = np.min(smpEIs)
                
                else:
                    # Evaluate the inactive regions by uniformly sampling 
                    smp = sample_from_discontinuous_region(configs['sampling']['num_inactive'], [reg], totalVolume, self.tf_dim, rng, volume=True ) #uniform_sampling(5, internal_inactive_subregion[0].input_space, self.tf_dim, self.rng)
                    mu, std = self._surrogate(reg.model, smp)
                    reward = float('inf')
                    for i in range(len(smp)):
                        f_xt = np.random.normal(mu[i],std[i],1)
                        smp_reward = self.costOfInactive(f_xt,ytr)
                        if reward > -1*smp_reward:
                            reward = ((-1 * smp_reward))
                    reg.rewardDist[ix] = (reward)
            
            agents = sorted(agents, key=lambda x: x.agentId)

        # Get the minimum encountered so far from the active regions based on predicted values
        f_ro = self.get_min_across_region_samples(agents)
        return f_ro, rl_root

    # Function to get the minimum across the set of active regions
    def get_min_across_regions(self, agents, tmpGP):
        minytrval = float('inf')
        minytr = []
        for ia in agents:
            try:
                minidx = tmpGP.dataset._getMinIdx(ia.obsIndices)
            except IndexError:
                exit(1)
            if minytrval > tmpGP.dataset.y_train[minidx]:
                minytrval = tmpGP.dataset.y_train[minidx]
                minytr = tmpGP.dataset.y_train[minidx] #min(minytrval, tmpGP.dataset.y_train[minidx])
        ytr = minytr

        return ytr
    
    def get_min_across_region_samples(self, agents):
        minytrval = float('inf')
        minytr = []
        for ia in agents:
            try:
                minytrval = min(minytrval, min(ia.yOfsmpIndices.values()))
            except AssertionError:
                exit(1)
        return minytrval
        
    # Cost calulation for inactive regions
    # Cost is the difference between the observed min and the obs from the posterior
    def costOfInactive(self,f_xt,ytr):
        ymin = np.min(ytr)
        r = max(ymin - f_xt, 0)
        # print(' each regret : ', r)
        return r
    
    # Cost function for active regions which is our base policy heuristic (EI)
    def cost(self,xt,ytr, model, sample_type='single'):
        r = self._acquisition(ytr, xt, model,sample_type)
        return r

# Function to rollout N^(RO) times 
def simulate(m, root, globalGP, mc_iters, num_agents, tf, tf_dim, behavior, horizon, rng):
    """Step 2: Simulate Configuration rollout

        Args: 
            m: ith agent to rollout fixing (1 to i-1) agent configurations
            root: Configuration to rollout
            mc_iters: Number of times to repeat the simulation (N^(RO))
            num_agents: Number of agents
            test_function_dimension: The dimensionality of the region. (Dimensionality of the test function)

        Return:
            F_nc: Average approximate Q-factor of the configuration
    """
    total_time = 0
    roll = RolloutEI()
    lvs = root.find_leaves()
    
    for l in lvs:
        l.state = State.ACTUAL
    root = saveRegionState(root)  # Save the original state of the tree
    f_nc = 0

    for r in range(mc_iters):
        print(f'= MC {r}'*50)
        start_time = time.time()  # Start time

        # Rollout the current configuration
        root, f_ro = roll.rollout(m, root, globalGP, horizon, num_agents, tf, tf_dim, behavior, rng)  # Execute the operation and build the tree
        
        # Sum up the min ecountered over N^(RO) times 
        f_nc += f_ro

        # Get actual leaves
        lvs = getActualState(root)
        for sima in lvs:
            # Accumulate the improvements across regions to make the actual jumps
            accumulate_rewardDist(sima, num_agents)
            sima.rewardDist = np.asarray(sima.rewardDist, dtype="object").reshape((1, num_agents))
            sima.avgRewardDist = np.vstack((sima.avgRewardDist, sima.rewardDist))
            sima.avgRewardDist = np.sum(sima.avgRewardDist, axis=0)
            

        end_time = time.time()  # End time
        total_time += end_time - start_time  
        # print(f'^ {r}'*50)
        # print_tree(root)
        # print('^'*50)
        # Restore region state
        numag=0
        for l in lvs:
            restoreRegionState(l, ['avgRewardDist','yOfsmpIndices']) #,'samples','smpIndices'])  
            # print('b4 state : ', l, l.input_space, root, root.input_space)
            l.state = State.ACTUAL
            saveRegionState(l)
            # print('l after obsinx :', l.obsIndices)
            if l.getStatus() == RegionState.ACTIVE.value:
                numag += 1
        assert numag == num_agents

        # print('^ after restore'*10)
        # print_tree(root, routine="MAIN")
        # print_tree(root)
        # exit(1)
    print('total time ', total_time)
    # exit(1)

    # Compute the average improvements across regions over N^(RO) times 
    for lf in lvs:
            lf.avgRewardDist = lf.avgRewardDist / mc_iters
            lf.rewardDist = lf.avgRewardDist
    # exit(1)
    # print('^ after mc '*10)
    # print_tree(root, routine="MAIN")
    # print()
    # print_tree(root)
    # Average the min ecountered over N^(RO) times 
    F_nc = f_nc/mc_iters
    print('F_NC', F_nc)
    return root, F_nc #, smpGP

