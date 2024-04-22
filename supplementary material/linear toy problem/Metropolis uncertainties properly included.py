import numpy as np
import pymc as pm
import arviz as az

#observed data (experimental measurements)
data = [163.1, 73.2, 115.7, 107.9]

#covariance matrix with propagated uncertainty from UMIP
cd = np.random.multivariate_normal(mean=[8,4], cov=[[1, 0.5],[0.5,1]], size=1000000)
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

"""
# correct covariance matrix, the same as "measurement_covariance" (within Monte Carlo uncertainty)
cov = np.array([[ 3.02539885,  4.52413505,  6.03136712,  3.01458095],
                [ 4.52413505,  7.0467328,   9.04926106,  4.52268433],
                [ 6.03136712,  9.04926106, 12.07370549,  6.02958629],
                [ 3.01458095,  4.52268433,  6.02958629,  3.02373706]])
# covariance matrix with uncertainties only propagated to the diagonal
cov = np.array([[ 3.02539885,  0,  0,  0],
                [ 0,  7.0467328,   0,  0],
                [ 0,  0, 12.07370549,  0],
                [ 0,  0,  0,  3.02373706]])
# uncertainties from uncalibrated model input parameters completely neglected
cov = np.array([[ 0.01,  0,  0,  0],
                [ 0,  0.01,   0,  0],
                [ 0,  0, 0.01,  0],
                [ 0,  0,  0,  0.01]])
"""

with pm.Model() as basic_model:
    # Priors for unknown model parameters
    a = pm.Normal("a", mu=35, sigma=5)
    b = pm.Normal("b", mu=15, sigma=3)
    # Mathematical models
    mu = [3*a+2*b+8+4, a+b+2*8+4, a+4*b+2*8+2*4, 2*a + b + 8+4]
    # Distribution of observed data, including propagated uncertainties
    Y_obs = pm.MvNormal("Y_obs", mu=mu, cov=measurement_covariance, observed=data)
    # Sampling
    trace = pm.sample(tune=1000, draws=45000, cores=1, chains=1, step=pm.Metropolis())

# Printing posterior means and standard deviations, 3 digit precision
print(az.summary(trace, kind="stats"))
# Plotting posterior
az.plot_trace(trace);
# Saving the posterior data
#az.to_netcdf(trace, '/home/jedrzejczykm/Desktop/paper/posterior_data_Metropolis.nc')