import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

# Assimilated data - experimental measurements
data = [163.1, 73.2, 115.7, 107.9]
# Random number generator, required for "Simulator"
rng = np.random.default_rng()

# Mathematical model of the assimilated data
def normal_sim(rng, a, b, size=1):
    # cd - the c and d variables modeled as correlated noise
    cd=rng.multivariate_normal([8, 4], np.array([[1, 0.5], [0.5, 1]]), 1)
    return [3*a+2*b+cd[0][0]+cd[0][1], a+b+2*cd[0][0]+cd[0][1], (a+4*b+2*cd[0][0]+2*cd[0][1]), 2*a + b + cd[0][0]+cd[0][1]]

with pm.Model() as example:
    # Priors for unknown model parameters
    a = pm.Normal("a", mu=35, sigma=5)
    b = pm.Normal("b", mu=15, sigma=3)
    # epsilon is the uncertainty of experimental measurements of assimilated data. 
    # If the uncertainty of experimental measurements varies, the data and normal_sim results 
    # would have to be rescaled. For example, if the third measurement (115.7) was taken with
    # uncertainty of 1, while all others were taken with uncertainty of 0.1, the data
    # would look as follows: data = [163.1, 73.2, 115.7/10, 107.9]
    # and mathematical model would look as follows:
    # 3*a+2*b+cd[0][0]+cd[0][1], a+b+2*cd[0][0]+cd[0][1], (a+4*b+2*cd[0][0]+2*cd[0][1])/10, 2*a + b + cd[0][0]+cd[0][1]
    s = pm.Simulator("s", normal_sim, params=(a, b), sum_stat="sort", epsilon=0.1, observed=data)
    # Sampling step. 50000 is the number of samples to draw from the posterior and 
    # the number of independent markov chains.
    # To obtain a smoother posterior increase this number from 50000 to a higher one
    idata = pm.sample_smc(50000, cores=1, chains=1, threshold=0.5)
    #sampling posterior values of data
    idata.extend(pm.sample_posterior_predictive(idata))

# Posterior of calibrated input parameters
az.plot_trace(idata, kind="rank_vlines");
print(az.summary(idata, kind="stats"))
# Posterior of simulated quantitites
#az.plot_trace(idata.posterior_predictive);
#print(az.summary(idata.posterior_predictive, kind="stats"))
# Saving the posterior data
#az.to_netcdf(idata, '/home/jedrzejczykm/Desktop/paper/posterior_data_SMC_ABC.nc')