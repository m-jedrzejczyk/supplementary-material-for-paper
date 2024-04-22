import pymc as pm
import numpy as np
from scipy.stats import gaussian_kde
import arviz as az
import matplotlib.pyplot as plt
rng = np.random.default_rng()

data = [1238.5, 215, 1544]

with pm.Model() as model:
    # Define priors for a and b
    a = pm.Uniform("a", lower=0.1*16, upper=3*16)
    b = pm.Uniform("b", lower=0.1*14, upper=3*14)
    c = 7
    d = 4
    mu=[3*a**2 + 2*b**2+c + d**2, a**(3/2) + b**2+2*c + d, a**(5/2) + 4*b**(3/2)+2*c + 2*d]
    Y_obs = pm.Normal("Y_obs", mu=[mu[0], mu[1], mu[2]], sigma=1, observed=data)
    trace = pm.sample(tune=2000, chains=1, draws=15000, cores=1, step=pm.NUTS())
    #trace.extend(pm.sample_posterior_predictive(trace))

#az.to_netcdf(trace, '/home/jedrzejczykm/Desktop/paper/NUTS IUQ 15000 unc neglected.nc')
print(az.summary(trace, kind="stats"))
az.plot_trace(trace, kind="rank_vlines");