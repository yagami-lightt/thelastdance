import pandas as pd
from scipy import stats

# Load dataset from CSV
df = pd.read_csv("twoSampledatset.csv")

# Check if there are exactly two columns for comparison
columns = df.columns
if len(columns) != 2:
    print("Error: Dataset must have exactly two columns for comparison.")
else:
    # Extract the two variables
    sample1 = df[columns[0]]
    sample2 = df[columns[1]]

    # Descriptive statistics
    mean1 = sample1.mean()
    mean2 = sample2.mean()
    variance1 = sample1.var()
    variance2 = sample2.var()

    # Checking if the samples are paired (same length)
    if len(sample1) == len(sample2):
        # Paired t-test (if lengths of both columns are equal)
        t_stat, p_value_two_tail = stats.ttest_rel(sample1, sample2)
        p_value_one_tail = p_value_two_tail / 2  # One-tailed p-value
        test_type = "Paired T-Test"
    else:
        # Two-sample t-test (independent samples) if the lengths are different
        t_stat, p_value_two_tail = stats.ttest_ind(sample1, sample2)
        p_value_one_tail = p_value_two_tail / 2  # One-tailed p-value
        test_type = "Two-Sample T-Test"

    # Degrees of freedom (for two-sample test)
    n1 = len(sample1)
    n2 = len(sample2)
    dfree = n1 + n2 - 2

    # Critical values for one-tail and two-tail
    alpha = 0.05
    t_critical_one_tail = stats.t.ppf(1 - alpha, dfree)
    t_critical_two_tail = stats.t.ppf(1 - alpha / 2, dfree)

    # Print results
    print(f"{test_type} Results:")
    print("Mean of Sample1:", mean1)
    print("Mean of Sample2:", mean2)
    print("Variance of Sample1:", variance1)
    print("Variance of Sample2:", variance2)
    print("Degrees of Freedom (df):", dfree)
    print("t-Statistic:", t_stat)
    print("P-value (Two-Tail):", p_value_two_tail)
    print("P-value (One-Tail):", p_value_one_tail)
    print("t Critical Value (One-Tail):", t_critical_one_tail)
    print("t Critical Value (Two-Tail):", t_critical_two_tail)

    # Conclusion based on p-value
    if p_value_two_tail < alpha:
        print("Conclusion: Reject null hypothesis (significant difference between the groups)")
    else:
        print("Conclusion: Fail to reject null hypothesis (no significant difference between the groups)")
