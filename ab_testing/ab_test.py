import statsmodels.stats.api as sms

# Variables
v1 = 0.1  # Baseline conversion rate
v2 = 0.12  # Target conversion rate (p1 + MDE)
alpha = 0.05
power = 0.9

# Calculate effect size
effect_size = sms.proportion_effectsize(v1, v2)
print(f"The effect size is: {effect_size}")

# Calculate sample size
sample_size = sms.NormalIndPower().solve_power(effect_size, power=power, alpha=alpha, ratio=1)
print(f"Required sample size per variant: {int(sample_size)}")

import numpy as np
from statsmodels.stats.proportion import proportions_ztest

# Example data
control_successes = 441  # Number of successes in the control group
control_total = sample_size     # Total number of trials in the control group
test_successes = 503     # Number of successes in the test group
test_total = sample_size        # Total number of trials in the test group

# Data for z-test
count = np.array([control_successes, test_successes])
nobs = np.array([control_total, test_total])

# Perform z-test
stat, p_value = proportions_ztest(count, nobs)
print(f"Z-statistic: {stat:.4f}, p-value: {p_value:.4f}")

# Interpretation
if p_value < alpha:
    print(f"Reject the null hypothesis. There is a significant difference between the two variants for a sample size of {int(sample_size)}")
else:
    print(f"Fail to reject the null hypothesis. There is no significant difference between the two variants for a sample size of {int(sample_size)}")
