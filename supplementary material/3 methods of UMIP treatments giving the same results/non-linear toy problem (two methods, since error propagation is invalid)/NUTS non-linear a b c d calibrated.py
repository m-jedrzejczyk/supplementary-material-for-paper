import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

data = [1238.5, 215, 1544]
rng = np.random.default_rng()


with pm.Model() as model:
    # Define priors for a, b, and sigma
    a = pm.Uniform("a", lower=0.1*16, upper=3*16)
    b = pm.Uniform("b", lower=0.1*14, upper=3*14)
    c = pm.Uniform("c", lower=5, upper=9)
    d = pm.Uniform("d", lower=2, upper=6)
    # Mathematical models
    mu=[3*a**2 + 2*b**2+c + d**2, a**(3/2) + b**2+2*c + d, a**(5/2) + 4*b**(3/2)+2*c + 2*d]
    # Distribution of observed data
    Y_obs = pm.Normal("Y_obs", mu=[mu[0], mu[1], mu[2]], sigma=1, observed=data)
    # Sampling
    idata = pm.sample(tune=2000, chains=1, draws=55000)
    # Calculating posterior of simulated values using posteriors of a,b,sigma
    idata.extend(pm.sample_posterior_predictive(idata))

#saving posterior data
az.to_netcdf(idata, 'NUTS 55000.nc')

az.plot_trace(idata, kind="rank_vlines");
print(az.summary(idata, kind="stats"))
az.plot_trace(idata.posterior_predictive);
print(az.summary(idata.posterior_predictive, kind="stats"))

a = idata.posterior.a.data[0]
b = idata.posterior.b.data[0]

stackab1 = np.vstack((a, b))
covab1 = np.cov(stackab1)
corrab1 = np.corrcoef(stackab1)

print("nuts posterior covariance:")
print(covab1)
corrab1 = np.corrcoef(stackab1)
print("nuts posterior correlation:")
print(corrab1)
