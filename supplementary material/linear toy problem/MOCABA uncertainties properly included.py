import numpy as np

np.random.seed(1000)
# Experimental measurements (data for assimilation)
experiments = np.array([163.6, 74.2, 116.7, 108.4])
# Sampling model input parameters designated for calibration
a = np.random.normal(35, 7, 1000)
b = np.random.normal(15, 5, 1000)
sampled_input_parameters = np.column_stack((a, b))
# Error propagation to experimental covariance matrix
cd = np.random.multivariate_normal(mean=[8,4], cov=[[0.5**2, 0.5**3], [0.5**3, 0.5**2]], size=1000)
measurement_error1=np.random.normal(0,0.1,1000)
measurement_error2=np.random.normal(0,0.1,1000)
measurement_error3=np.random.normal(0,0.1,1000)
measurement_error4=np.random.normal(0,0.1,1000)
e1 = cd[:, 0] + cd[:, 1] + measurement_error1
e2 = 2*cd[:, 0] + cd[:, 1] + measurement_error2
e3 = 2*cd[:, 0] + 2*cd[:, 1] + measurement_error3
e4 = cd[:, 0] + cd[:, 1] + measurement_error4
stack_error_propagation = np.vstack((e1, e2, e3, e4))
measurement_covariance = np.cov(stack_error_propagation)

# Simulating results using sampled values of "a" and "b"
x1 = 3*a+2*b+8+4  
x2 = a+b+2*8+4
x3 = a+4*b+2*8+2*4
x4 = 2*a+b+8+4
simulated_results = np.vstack((x1, x2, x3, x4)).T
#calculating prior means
prior_mean = np.zeros(4)
prior_mean[0] = np.mean(x1)
prior_mean[1] = np.mean(x2)
prior_mean[2] = np.mean(x3)
prior_mean[3] = np.mean(x4)

# Joining the "a", "b" samples with corresponding simulated results
prior_cov = np.cov(simulated_results.T, sampled_input_parameters.T)

# Number of experiments
number_of_experiments=4
#preparing matrices for calculations. Naming in accordance with nomenclature from Hoefer, Buss 2015
covAB = prior_cov[0:number_of_experiments, number_of_experiments:]
covB = prior_cov[0:number_of_experiments, 0:number_of_experiments]
covVB = measurement_covariance[0:number_of_experiments, 0:number_of_experiments]
vB = experiments[0:number_of_experiments]
covA = prior_cov[number_of_experiments:,number_of_experiments:]

prior_appl_mean = np.array([np.mean(a), np.mean(b)])
# Performing the Bayesian updating
posterior_appl_mean = prior_appl_mean + np.dot(covAB.T, np.dot(np.linalg.inv(covB+covVB), (vB-prior_mean)))
posterior_appl_cov = covA - np.dot(covAB.T, np.dot(np.linalg.inv(covB+covVB), covAB))
posterior_appl_stds = np.sqrt(np.diag(posterior_appl_cov))
posterior_benchcov_mean = covB-np.dot(covB.T, np.dot(np.linalg.inv(covB+covVB), covB))
posterior_bench_mean = prior_mean + np.dot(covB.T, np.dot(np.linalg.inv(covB+covVB), (vB-prior_mean)))

# Print posterior means and standard deviations
print(posterior_appl_mean)
print(np.sqrt(np.diag(posterior_appl_cov)))
