import pymc as pm
import numpy as np
from scipy.stats import gaussian_kde
import arviz as az
import matplotlib.pyplot as plt
rng = np.random.default_rng()

#observed data (experimental measurements)
data = [1238.5, 215, 1544]

with pm.Model() as model:
    # Define priors for a, b, and sigma
    a = pm.Uniform("a", lower=0.1*16, upper=3*16)
    b = pm.Uniform("b", lower=0.1*14, upper=3*14)
    sigma = pm.Uniform("sigma", lower=0.1, upper=5)
    c = 7
    d = 4
    # Mathematical models
    mu=[3*a**2 + 2*b**2+c + d**2, a**(3/2) + b**2+2*c + d, a**(5/2) + 4*b**(3/2)+2*c + 2*d]
    # Distribution of observed data
    Y_obs = pm.Normal("Y_obs", mu=[mu[0], mu[1], mu[2]], sigma=(1+sigma**2)**(1/2), observed=data)
    # Sampling. "target accept" is a parameter that takes value of 0.8 by default, but 
    # needs increasing if divergences appear during calculations
    trace = pm.sample(tune=2000, chains=1, cores=1, draws=15000, nuts={'target_accept':0.92})
    # Calculating posterior of simulated values using posteriors of a,b,sigma
    trace.extend(pm.sample_posterior_predictive(trace))

#az.to_netcdf(trace, '/home/jedrzejczykm/Desktop/papers/NUTS IUQ 15000 unc as sigma.nc')
print(az.summary(trace, kind="stats"))
#az.plot_trace(trace.posterior_predictive, kind="rank_vlines");
#print(az.summary(trace.posterior_predictive, kind="stats"))

