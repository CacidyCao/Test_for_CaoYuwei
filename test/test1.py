# -*- coding: utf-8 -*-
# @Time : 2024/1/21 21:13
# @Author : cyw
# @File : test1.py
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import statsmodels.api as sm


# Define a function to simulate the AR(1) process
def simulate_ar1(phi, sigma, N):
    """ a basic simulate function

    @param phi:
    @param sigma:
    @param N:
    @return:
    """
    np.random.seed(0)
    errors = np.random.normal(0, sigma, N)
    X = np.zeros(N)
    for t in range(1, N):
        X[t] = phi * X[t - 1] + errors[t]
    return X


# question 1.1
# Load parameters from the .json file
with open('parameters.json', 'r') as file:
    params = json.load(file)

# Unpack the parameters
phi = params['phi']
sigma = params['sigma']
N = params['N']

# Define custom business day frequency
business_day = pd.tseries.offsets.CustomBusinessHour(start="09:30", end="15:00")

# Simulate the AR(1) process
start_date = pd.Timestamp("2000-03-06 09:30:00")
dates = pd.date_range(start=start_date, periods=N, freq=business_day)
X = simulate_ar1(phi, sigma, N)

# Create a pandas DataFrame with timestamps as the index
df1 = pd.DataFrame(X, index=dates, columns=['Simulated_Time_Series'])
# df.to_csv('simulated_time_series.csv')

# question 1.2
num_simulations = 20
# Initialize a 2D array to store the simulations
X = np.zeros((num_simulations, N))

# Simulate the AR(1) process for each simulation
for i in range(num_simulations):
    X[i, :] = simulate_ar1(phi, sigma, N)

# Create a pandas DataFrame for each simulation
dfs = [pd.DataFrame(X[i], index=dates, columns=[f'Simulation_{i + 1}_Time_Series']) for i in range(num_simulations)]

colors = plt.cm.viridis(np.linspace(0, 1, len(dfs)))
plt.figure(figsize=(12, 8))

for df, color in zip(dfs, colors):
    plt.plot(df.index, df.values, color=color)
plt.xlabel('Datetime')
plt.ylabel('Simulated Values')
plt.title('Simulated AR(1) Time Series')
plt.legend()
plt.show()

# question 1.3
# Estimate ARIMA parameters for each simulation
model = sm.tsa.ARIMA(df1, order=(1, 0, 0))  # Assuming AR(1) process
results = model.fit()

# True parameters from the .json file
true_phi = phi
true_sigma = sigma

# Estimated parameters
estimated_phi = results.arparams[0]
estimated_sigma = np.sqrt(np.var(results.resid))  # Calculating the variance from the residuals

# Print true and estimated parameters for each simulation
print(f"Simulation {i + 1}:")
print(f"True Phi: {true_phi}, True Sigma: {true_sigma}")
print(f"Estimated Phi: {estimated_phi}, Estimated Sigma: {estimated_sigma}")
print("\n")

# List of N values to test
N_values = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]
estimated_phis = []
for N in N_values:
    simulated_data = simulate_ar1(true_phi, true_sigma, N)

    # Estimate AR(1) parameters
    model = sm.tsa.ARIMA(simulated_data, order=(1, 0, 0))
    fitted_model = model.fit()
    estimated_phi = fitted_model.arparams[0]
    estimated_phis.append(estimated_phi)

# Compare estimated parameters with the true parameter value
plt.plot(N_values, [true_phi] * len(N_values), label='True Phi')
plt.plot(N_values, estimated_phis, marker='o', linestyle='-', label='Estimated Phi')
plt.xscale('log')  # Log scale for x-axis
plt.xlabel('Number of Timesteps (N)')
plt.ylabel('Parameter Value')
plt.title('Impact of Increasing N on Parameter Estimation')
plt.legend()
plt.show()
