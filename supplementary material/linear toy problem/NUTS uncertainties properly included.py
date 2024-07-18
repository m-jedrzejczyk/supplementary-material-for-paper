import numpy as np
import pymc as pm
import arviz as az

#observed data (experimental measurements)
data = [163.6, 74.2, 116.7, 108.4]
#covariance matrix with propagated uncertainty from UMIP
cd = np.random.multivariate_normal(mean=[8,4], cov=[[0.5**2, 0.5**3], [0.5**3, 0.5**2]], size=1000000)
#cd = np.random.multivariate_normal(mean=[8,4], cov=[[1, 0.5],[0.5,1]], size=1000000)
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

measurement_covariance = np.array([[ 0.01,  0,  0,  0],
                [ 0,  0.01,   0,  0],
                [ 0,  0, 0.01,  0],
                [ 0,  0,  0,  0.01]])
"""
# correct covariance matrix, the same as "measurement_covariance" (within Monte Carlo uncertainty)
cov = np.array([[0.76025151, 1.12510038, 1.50052684, 0.75022842],
                [1.12510038, 1.75971513, 2.25022741, 1.12506701],
                [1.50052684, 2.25022741, 3.01108469, 1.50051906],
                [0.75022842, 1.12506701, 1.50051906, 0.76024915]])
# covariance matrix with uncertainties only propagated to the diagonal
cov = np.array([[ 0.76025151,  0,  0,  0],
                [ 0,  1.75971513,   0,  0],
                [ 0,  0, 3.01108469,  0],
                [ 0,  0,  0,  0.76024915]])
# uncertainties from uncalibrated model input parameters completely neglected
cov = np.array([[ 0.01,  0,  0,  0],
                [ 0,  0.01,   0,  0],
                [ 0,  0, 0.01,  0],
                [ 0,  0,  0,  0.01]])
"""

with pm.Model() as basic_model:
    # Priors for unknown model parameters
    a = pm.Normal("a", mu=35, sigma=7)
    b = pm.Normal("b", mu=15, sigma=5)
    # Mathematical models
    mu = [3*a+2*b+8+4, a+b+2*8+4, a+4*b+2*8+2*4, 2*a + b + 8+4]
    # Distribution of observed data, including propagated uncertainties
    Y_obs = pm.MvNormal("Y_obs", mu=mu, cov=measurement_covariance, observed=data)
    # Sampling
    trace = pm.sample(tune=1000, draws=5000, cores=1, chains=1, step=pm.NUTS())
    trace.extend(pm.sample_posterior_predictive(trace))

# Printing posterior means and standard deviations, 3 digit precision
print(az.summary(trace, kind="stats"))
# Plotting posterior
az.plot_trace(trace);
# plotting posterior predictive
# az.plot_trace(trace.posterior_predictive, kind="rank_vlines");
# print(az.summary(trace.posterior_predictive, kind="stats"))
# Saving the posterior data
az.to_netcdf(trace, '/home/jedrzejczykm/Desktop/my_papers/paper3/files for R1/trace_uncertainty_fully_neglected_NUTS_5000')
