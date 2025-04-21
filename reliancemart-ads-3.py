import pandas as pd
from scipy import stats

# Load dataset from CSV
df = pd.read_csv("reliancemart.csv")

# Extract Rice_Bag_Weight column
weights = df["Rice_Bag_Weight"]

# Parameters
hypothesized_mean = 25
n = len(weights)  # Number of observations
dfree = n - 1  # Degrees of freedom

# 1. Descriptive Statistics
mean = weights.mean()  # Mean
variance = weights.var()  # Variance
std_dev = weights.std()  # Standard Deviation
observations = n  # Number of observations

# 2. Paired t-test (One-sample t-test for hypothesized mean)
t_stat, p_value_two_tail = stats.ttest_1samp(weights, hypothesized_mean)
p_value_one_tail = p_value_two_tail / 2  # One-tailed p-value

# 3. t Critical Values for one-tail and two-tail
alpha = 0.05
t_critical_one_tail = stats.t.ppf(1 - alpha, dfree)  # One-tail critical value
t_critical_two_tail = stats.t.ppf(1 - alpha / 2, dfree)  # Two-tail critical value

# 4. Pearson Correlation (only works for two variables, so skip here)
# In this case, we don’t have another variable, so we won’t compute correlation

# Print the results
print("Mean:", mean)
print("Variance:", variance)
print("Standard Deviation:", std_dev)
print("Number of Observations:", observations)
print("Hypothesized Mean Difference:", hypothesized_mean)
print("Degrees of Freedom (df):", dfree)
print("t-Statistic:", t_stat)
print("P-value (Two-Tail):", p_value_two_tail)
print("P-value (One-Tail):", p_value_one_tail)
print("t Critical Value (One-Tail):", t_critical_one_tail)
print("t Critical Value (Two-Tail):", t_critical_two_tail)

# Conclusion based on p-value
if p_value_two_tail < alpha:
    print("Conclusion: Reject null hypothesis (significant difference from 25kg)")
else:
    print("Conclusion: Fail to reject null hypothesis (no significant difference from 25kg)")
