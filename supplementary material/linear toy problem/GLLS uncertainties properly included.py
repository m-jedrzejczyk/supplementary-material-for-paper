import numpy as np
from scipy.stats import multivariate_normal
from numpy.linalg import inv

# True parameter values
a_true, b_true, c_true, d_true = 42, 13, 7, 4

# Prior means and standard deviations for a, b
prior_means = np.array([35, 15], dtype=np.float64)
prior_std_devs = np.array([5, 3], dtype=np.float64)
prior_covariance = np.diag(prior_std_devs**2)
# Experimental measurements - assimilated data
observed_data = np.array([163.1, 73.2, 115.7, 107.9])

# Error propagation of c,d to experimental covariance matrix
# To investigate improper treatment of uncalibrated model input parameters modify the covariance matrix
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

# Define the model equations
def model(parameters):
    a, b = parameters
    x1 = 3*a + 2*b + 8 + 4
    x2 = a + b + 2*8 + 4
    x3 = a + 4*b + 2*8 + 2*4
    x4 = 2*a + b + 8 + 4
    return np.array([x1, x2, x3, x4])

# Bayesian update
def bayesian_update(prior_means, prior_covariance, observed_data, measurement_covariance):
    # Predicted data based on prior means
    predicted_data = model(prior_means)
    # Here the sensitivities of simulated data to calibrated model input parameter changes are defined
    J = np.array([
        [3, 2],
        [1, 1],
        [1, 4],
        [2, 1]
    ])
    # Auxiliary calculations for the update step:
    # Discrepancy vector
    d = predicted_data-observed_data
    # Covariance matrix of the predicted_data
    C_kk = J @ prior_covariance @ J.T
    #  Uncertainty matrix for the discrepancy vector d
    C_dd = C_kk + measurement_covariance
    # Vector containing difference between prior input parameters and posterior input parameters
    delta_alpha = - (prior_covariance @ J.T @ np.linalg.inv(C_dd)) @ d
    
    # Update step
    posterior_covariance = prior_covariance - prior_covariance @ J.T @ np.linalg.inv(C_dd) @ J @ prior_covariance
    posterior_means = prior_means + delta_alpha
    return posterior_means, posterior_covariance

posterior_means, posterior_covariance = bayesian_update(prior_means, prior_covariance, observed_data, measurement_covariance)
print("Posterior Means: ", posterior_means)
print("Posterior Standard Deviations: ", np.sqrt(np.diag(posterior_covariance)))
print(posterior_covariance)
