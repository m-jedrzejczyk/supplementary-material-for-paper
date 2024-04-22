import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

data = [1238.5, 215, 1544]
rng = np.random.default_rng()

def normal_sim(rng, a, b, sig, size=1):
    c = rng.uniform(5, 9, 1)
    d = rng.uniform(2, 6, 1)
    return  [3*a*a+2*b*b+c[0]+d[0]**2, a**(3/2)+b*b+2*c[0]+d[0], a**(5/2)+4*b**(3/2)+2*c[0]+2*d[0]]

with pm.Model() as example:
    a = pm.Uniform("a", lower=0.1*16, upper=3*16)
    b = pm.Uniform("b", lower=0.1*14, upper=3*14)
    s = pm.Simulator("s", normal_sim, params=(a, b), sum_stat="sort", epsilon=1, observed=data)
    idata = pm.sample_smc(85000, cores=1, chains=1, threshold=0.5)
    idata.extend(pm.sample_posterior_predictive(idata))

#saving posterior data
az.to_netcdf(idata, 'SMC-ABC 85000.nc')

az.plot_trace(idata, kind="rank_vlines");
print(az.summary(idata, kind="stats"))
az.plot_trace(idata.posterior_predictive);
print(az.summary(idata.posterior_predictive, kind="stats"))

a = idata.posterior.a.data[0]
b = idata.posterior.b.data[0]

stackab = np.vstack((a, b))
covab = np.cov(stackab)
print("smc abc posterior covariance:")
print(covab)
corrab = np.corrcoef(stackab)
print("smc abc posterior correlation:")
print(corrab)
