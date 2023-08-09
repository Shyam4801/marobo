RED = '\033[91m'
GREEN = '\033[92m'
END = '\033[0m'


import pandas as pd
import numpy as np

# print(np.random.multinomial(10, [0.2,0.5,0.3]))
def internal_function(X, from_agent = None): #Branin with unique glob min -  9.42, 2.475 local min (3.14, 12.27) and (3.14, 2.275)
            x1 = X[0]
            x2 = X[1]
            t = 1 / (8 * np.pi)
            s = 10
            r = 6
            c = 5 / np.pi
            b = 5.1 / (4 * np.pi ** 2)
            a = 1
            term1 = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2
            term2 = s * (1 - t) * np.cos(x1)
            l1 = 5 * np.exp(-5 * ((x1+3.14)**2 + (x2-12.27)**2))
            l2 = 5 * np.exp(-5 * ((x1+3.14)**2 + (x2-2.275)**2))
            return term1 + term2 + s + l1 + l2

glob_mins = np.array([[9.42, 2.475],[3.14, 12.27],[3.14, 2.275]])
y_of_mins = np.array([internal_function(i) for i in glob_mins])
print(y_of_mins)

import numpy as np
from scipy.optimize import minimize

# Define the Three-Hump Camel Function
def three_hump_camel(x):
    return 2 * x[0]**2 - 1.05 * x[0]**4 + (x[0]**6 / 6) + x[0] * x[1] + x[1]**2

# Find the global minimum using numerical optimization
result_global = minimize(three_hump_camel, [0, 0], method='BFGS')

# Find the local minima using numerical optimization
result_local1 = minimize(three_hump_camel, [-0.27, 0.9235], method='BFGS')
result_local2 = minimize(three_hump_camel, [0.27, -0.9235], method='BFGS')

# Print the results
print("Global Minimum:")
print("Coordinates:", result_global.x)
print("Function Value:", result_global.fun)

print("\nLocal Minimum 1:")
print("Coordinates:", result_local1.x)
print("Function Value:", result_local1.fun)

print("\nLocal Minimum 2:")
print("Coordinates:", result_local2.x)
print("Function Value:", result_local2.fun)


# df = pd.read_csv('results/himmelblau50_rollout/himmelblau_5_50rollout.csv')


# df['ysofar'] = [min(df['y'].iloc[:i]) for i in range(1,len(df)+1)]
# print(df)

# import matplotlib.pyplot as plt

# plt.plot(df.index,df['ysofar'])
# # plt.show()
# plt.savefig('results/himmelblau50_rollout/himmelblau50_ysofar.png')


# def logdf(data,init_samp,maxbud, name, yofmins, rollout=False):
#     df = pd.DataFrame(np.array(data.history))
#     # df = df.iloc[:,1].apply(lambda x: x[0])
#     print(df)
#     print('_______________________________')
#     print('yofmins :',yofmins)
#     xcoord = pd.DataFrame(df.iloc[:,1].to_list())
#     xcoord['y'] = df.iloc[:,2]
#     xcoord['ysofar'] = [min(xcoord['y'].iloc[:i]) for i in range(1,len(xcoord)+1)]
#     if rollout:
#         rl='rollout'
#     else:
#         rl = 'n'
#     xcoord.to_csv('results/'+str(name)+'_'+str(init_samp)+'_'+str(maxbud)+rl+'.csv')
#     # plot_convergence(xcoord, name+str(maxbud)+'_'+rl)
#     plt.plot(df.index,df['ysofar'])
    
#     plt.savefig('results/'+str(name)+'_'+str(init_samp)+'_'+str(maxbud)+rl+'.png')
#     plt.show()
    # xcoord = xcoord.to_numpy()
    # print('_______________ Min Observed ________________')
    # print(xcoord[np.argmin(xcoord[:,2]), :])
    
    # return xcoord[np.argmin(xcoord[:,2]), :]