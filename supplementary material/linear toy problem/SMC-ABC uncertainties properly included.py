import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

# Assimilated data - experimental measurements
data = [163.6, 74.2, 116.7, 108.4]
# Random number generator, required for "Simulator"
rng = np.random.default_rng()

# Mathematical model of the assimilated data
def normal_sim(rng, a, b, size=1):
    cd=rng.multivariate_normal([8, 4], np.array([[0.5**2, 0.5**3], [0.5**3, 0.5**2]]), 1)
    return [3*a+2*b+cd[0][0]+cd[0][1], a+b+2*cd[0][0]+cd[0][1], (a+4*b+2*cd[0][0]+2*cd[0][1]), 2*a + b + cd[0][0]+cd[0][1]]

with pm.Model() as example:
    # Priors for unknown model parameters
    a = pm.Normal("a", mu=35, sigma=7)
    b = pm.Normal("b", mu=15, sigma=5)
    # epsilon is the uncertainty of experimental measurements of assimilated data. 
    # If the uncertainty of experimental measurements varies, the data and normal_sim results 
    # would have to be rescaled. For example, if the third measurement (116.7) was taken with
    # uncertainty of 1, while all others were taken with uncertainty of 0.1, the data and 
    # mathematical model would look as follows: data = [163.6, 74.2, 116.7/10, 108.4],
    # [3*a+2*b+cd[0][0]+cd[0][1], a+b+2*cd[0][0]+cd[0][1], (a+4*b+2*cd[0][0]+2*cd[0][1])/10, 2*a + b + cd[0][0]+cd[0][1]]
    s = pm.Simulator("s", normal_sim, params=(a, b), sum_stat="sort", epsilon=0.1, observed=data)
    idata = pm.sample_smc(95000, cores=1, chains=1, threshold=0.5)
    #sampling posterior values of data
    idata.extend(pm.sample_posterior_predictive(idata))

# Posterior of calibrated input parameters
az.plot_trace(idata, kind="rank_vlines");
print(az.summary(idata, kind="stats"))
# Posterior of simulated quantitites
az.plot_trace(idata.posterior_predictive);
print(az.summary(idata.posterior_predictive, kind="stats"))
# Saving the posterior data
#az.to_netcdf(idata, '/home/jedrzejczykm/Desktop/my_papers/paper3/toy_models/trace_uncertainty_included_SMC-ABC_200k.nc')
