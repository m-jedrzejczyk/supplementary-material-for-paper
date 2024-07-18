import numpy as np
import pymc as pm
import arviz as az

#observed data (experimental measurements)
data = [163.6, 74.2, 116.7, 108.4]

cd = np.random.multivariate_normal(mean=[8,4], cov=[[0.5**2, 0.5**3], [0.5**3, 0.5**2]], size=1000000)
measurement_error1=np.random.normal(0,0.1,1000000)
measurement_error2=np.random.normal(0,0.1,1000000)
measurement_error3=np.random.normal(0,0.1,1000000)
measurement_error4=np.random.normal(0,0.1,1000000)
e1 = cd[:, 0] + cd[:, 1] + measurement_error1
e2 = 2*cd[:, 0] + cd[:, 1] + measurement_error2
e3 = 2*cd[:, 0] + 2*cd[:, 1] + measurement_error3
e4 = cd[:, 0] + cd[:, 1] + measurement_error4
stack_error_propagation = np.vstack((e1, e2, e3, e4))
measurement_covariance = np.cov(stack_error_propagation)


with pm.Model() as basic_model:
    # Priors for unknown model parameters
    a = pm.Normal("a", mu=35, sigma=7)
    b = pm.Normal("b", mu=15, sigma=5)
    # Mathematical models
    mu = [3*a+2*b+8+4, a+b+2*8+4, a+4*b+2*8+2*4, 2*a + b + 8+4]
    # Distribution of observed data, including propagated uncertainties
    Y_obs = pm.MvNormal("Y_obs", mu=mu, cov=measurement_covariance, observed=data)
    # Sampling
    idata = pm.sample(tune=1000, draws=55000, cores=1, chains=1, step=pm.NUTS())

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
print("nuts_posterior cov:")
print(covab)
print("nuts_posterior corr:")
print(corrab)
