#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
get_ipython().run_line_magic('matplotlib', 'inline')


from mpl_toolkits.mplot3d import Axes3D
import numba

kappa = 10


# # Target Function
# 
# Lets create a target 1-D function with multiple local maxima to test and visualize how the [BayesianOptimization](https://github.com/fmfn/BayesianOptimization) package works. The target function we will try to maximize is the following:
# 
# $$f(x) = e^{-(x - 2)^2} + e^{-\frac{(x - 6)^2}{10}} + \frac{1}{x^2 + 1}, $$ its maximum is at $x = 2$ and we will restrict the interval of interest to $x \in (-2, 10)$.
# 
# Notice that, in practice, this function is unknown, the only information we have is obtained by sequentialy probing it at different points. Bayesian Optimization works by contructing a posterior distribution of functions that best fit the data observed and chosing the next probing point by balancing exploration and exploitation.

# In[2]:


def target(x):
    return np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/1) 

def target1(x, y):
    return 2*np.exp(-abs(x-1)) + 2*np.exp(-abs(y-1))


# In[3]:


x = np.linspace(-2, 10, 10000).reshape(-1, 1)
y = target(x)

plt.plot(x, y);


# 3d case
# =====================

xx = np.linspace(0, 2, 1000) 
yy = np.linspace(0, 2, 1000) 
Z = np.zeros((1000*1000))
meshX, meshY = np.meshgrid(xx, yy)

cnt = 0
for i in range(1000):
    for j in range(1000):
        Z[cnt] = target1(xx[j] , yy[i])
        cnt += 1
    


# Plot original data
fig = plt.figure()  
ax = Axes3D(fig)
ax.plot_surface(meshX, meshY, Z.reshape(-1, 1000),cmap='rainbow')
plt.title('test fun')
plt.xlabel('x'); plt.ylabel('y');  ax.set_zlabel('z')


# # Create a BayesianOptimization Object
# 
# Enter the target function to be maximized, its variable(s) and their corresponding ranges. A minimum number of 2 initial guesses is necessary to kick start the algorithms, these can either be random or user defined.

# In[4]:


optimizer = BayesianOptimization(target, {'x': (-2, 10)}, random_state=2)

optimizer1 = BayesianOptimization(target1, {'x': (0, 2), 'y':(0,2)}, random_state=27)


# In this example we will use the Upper Confidence Bound (UCB) as our utility function. It has the free parameter
# $\kappa$ which control the balance between exploration and exploitation; we will set $\kappa=5$ which, in this case, makes the algorithm quite bold.

# In[5]:


optimizer.maximize(init_points=3, n_iter=0, kappa = kappa)

optimizer1.maximize(init_points=2, n_iter=100, kappa=5)


# # Plotting and visualizing the algorithm at each step

# ### Let's first define a couple functions to make plotting easier

# In[6]:


def posterior(optimizer, x_obs, y_obs, grid):
    optimizer._gp.fit(x_obs, y_obs)

    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma

def plot_gp(optimizer, x, y):
    fig = plt.figure(figsize=(16, 10))
    steps = len(optimizer.space)
    fig.suptitle(
        'Gaussian Process and Utility Function After {} Steps'.format(steps),
        fontdict={'size':30}
    )
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
    
    x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])
    
    mu, sigma = posterior(optimizer, x_obs, y_obs, x)
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x, mu, '--', color='k', label='Prediction')

    axis.fill(np.concatenate([x, x[::-1]]), 
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.6, fc='c', ec='None', label='95% confidence interval')
    
    axis.set_xlim((-2, 10))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':20})
    axis.set_xlabel('x', fontdict={'size':20})
    
    utility_function = UtilityFunction(kind="ucb", kappa = kappa, xi=0)
    utility = utility_function.utility(x, optimizer._gp, 0)
    acq.plot(x, utility, label='Utility Function', color='purple')
    acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    acq.set_xlim((-2, 10))
    acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylabel('Utility', fontdict={'size':20})
    acq.set_xlabel('x', fontdict={'size':20})
    
    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    
    
    
def plot_gp1(optimizer, x, y, z):
    fig = plt.figure(figsize = (16,9))
    steps = len(optimizer.space)
    fig.suptitle(
        'Gaussian Process and Utility Function After {} Steps'.format(steps),
        fontdict={'size':30}
    )
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
    
    x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    y_obs = np.array([[res["params"]["y"]] for res in optimizer.res])
    z_obs = np.array([res["target"] for res in optimizer.res])
    
    mu, sigma = posterior(optimizer, x_obs, y_obs, x)
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x, mu, '--', color='k', label='Prediction')

    axis.fill(np.concatenate([x, x[::-1]]), 
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.6, fc='c', ec='None', label='95% confidence interval')
    
    axis.set_xlim((-2, 10))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':20})
    axis.set_xlabel('x', fontdict={'size':20})
    
    utility_function = UtilityFunction(kind="ucb", kappa=5, xi=0)
    utility = utility_function.utility(x, optimizer._gp, 0)
    acq.plot(x, utility, label='Utility Function', color='purple')
    acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    acq.set_xlim((-2, 10))
    acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylabel('Utility', fontdict={'size':20})
    acq.set_xlabel('x', fontdict={'size':20})
    
    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)


# ### Two random points
# 
# After we probe two points at random, we can fit a Gaussian Process and start the bayesian optimization procedure. Two points should give us a uneventful posterior with the uncertainty growing as we go further from the observations.

# In[7]:


plot_gp(optimizer, x, y)


# ### After one step of GP (and two random points)

# In[8]:


optimizer.maximize(init_points=0, n_iter=1, kappa = kappa)
plot_gp(optimizer, x, y)


# ### After two steps of GP (and two random points)

# In[9]:


optimizer.maximize(init_points=0, n_iter=1, kappa = kappa)
plot_gp(optimizer, x, y)


# ### After three steps of GP (and two random points)

# In[10]:


optimizer.maximize(init_points=0, n_iter=1, kappa = kappa)
plot_gp(optimizer, x, y)


# ### After four steps of GP (and two random points)

# In[11]:


optimizer.maximize(init_points=0, n_iter=1, kappa = kappa)
plot_gp(optimizer, x, y)


# ### After five steps of GP (and two random points)

# In[12]:


optimizer.maximize(init_points=0, n_iter=1, kappa = kappa)
plot_gp(optimizer, x, y)


# ### After six steps of GP (and two random points)

# In[13]:


optimizer.maximize(init_points=0, n_iter=1, kappa = kappa)
plot_gp(optimizer, x, y)


# ### After seven steps of GP (and two random points)

# In[14]:


optimizer.maximize(init_points=0, n_iter=1, kappa = kappa)
plot_gp(optimizer, x, y)


# # Stopping
# 
# After just a few points the algorithm was able to get pretty close to the true maximum. It is important to notice that the trade off between exploration (exploring the parameter space) and exploitation (probing points near the current known maximum) is fundamental to a succesful bayesian optimization procedure. The utility function being used here (Upper Confidence Bound - UCB) has a free parameter $\kappa$ that allows the user to make the algorithm more or less conservative. Additionally, a the larger the initial set of random points explored, the less likely the algorithm is to get stuck in local minima due to being too conservative.
