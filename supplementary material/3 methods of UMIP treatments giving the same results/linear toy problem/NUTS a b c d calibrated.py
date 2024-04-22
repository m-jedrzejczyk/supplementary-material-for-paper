import numpy as np
import pymc as pm
import arviz as az

#observed data (experimental measurements)
data = [163.1, 73.2, 115.7, 107.9]

with pm.Model() as basic_model:
    # Priors for unknown model parameters
    a = pm.Normal("a", mu=35, sigma=5)
    b = pm.Normal("b", mu=15, sigma=3)
    cd = pm.MvNormal("cd", mu=[8,4], cov=np.array([[1, 0.5], [0.5, 1]]))
    mu = [3*a+2*b+cd[0]+cd[1], a+b+2*cd[0]+cd[1], (a+4*b+2*cd[0]+2*cd[1]), 2*a + b + cd[0]+cd[1]]
    # Distribution of observed data, including propagated uncertainties
    Y_obs = pm.Normal("Y_obs", mu=mu, sigma=0.1, observed=data)
    # Sampling
    idata = pm.sample(tune=1000, draws=35000, cores=1, chains=1, step=pm.NUTS())

# Printing posterior means and standard deviations, 3 digit precision
print(az.summary(idata, kind="stats"))
# Plotting posterior
az.plot_trace(idata);

# Saving the posterior data
#az.to_netcdf(trace, '/home/jedrzejczykm/Desktop/my_papers/paper3/toy_models/trace_uncertainty_fully_neglected_NUTS_5000by4.nc')

a = idata.posterior.a.data[0]
b = idata.posterior.b.data[0]

stackab = np.vstack((a, b))
covab = np.cov(stackab)
corrab = np.corrcoef(stackab)
print("nuts all calibrated cov:")
print(covab)
print("nuts all calibrated corr:")
print(corrab)
