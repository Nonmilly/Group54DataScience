# stats_helpers.py
# shared stats functions so the task file stays clean and short.
# every function takes a sample (list or pandas Series) and returns a dict.

import math
import numpy as np
from scipy import stats


def describe(sample):
    """central tendency + dispersion for one set of numbers"""
    s = np.asarray(sample, dtype=float)
    s = s[~np.isnan(s)]
    q1 = float(np.percentile(s, 25))
    q3 = float(np.percentile(s, 75))
    return {
        "n": int(len(s)),
        "mean": float(np.mean(s)),
        "median": float(np.median(s)),
        "min": float(np.min(s)),
        "max": float(np.max(s)),
        "range": float(np.max(s) - np.min(s)),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "variance": float(np.var(s, ddof=1)),   # ddof=1 = sample variance
        "std": float(np.std(s, ddof=1)),         # sample standard deviation
    }


def check_normality(sample):
    """
    A rough check that the sample is roughly normal, for the conditions of
    inference. We look at skewness (near 0 = symmetric) and compare the
    mean and median (close together = symmetric).
    """
    s = np.asarray(sample, dtype=float)
    s = s[~np.isnan(s)]
    skew = float(stats.skew(s))
    return {
        "skewness": skew,
        "mean": float(np.mean(s)),
        "median": float(np.median(s)),
        # a common rule of thumb: |skewness| < 0.5 is fairly symmetric
        "roughly_normal": abs(skew) < 0.5,
    }


def confidence_interval(sample, conf=0.95):
    """
    Confidence interval for the population mean.
    Following the notes: use z* when the sample is large (n >= 30),
    use t* when it is small (n < 30).
    """
    s = np.asarray(sample, dtype=float)
    s = s[~np.isnan(s)]
    n = len(s)
    mean = float(np.mean(s))
    sd = float(np.std(s, ddof=1))
    standard_error = sd / math.sqrt(n)

    if n >= 30:
        # large sample -> z*
        critical = float(stats.norm.ppf((1 + conf) / 2))
        stat_name = "z*"
    else:
        # small sample -> t* with n-1 degrees of freedom
        critical = float(stats.t.ppf((1 + conf) / 2, n - 1))
        stat_name = "t*"

    margin = critical * standard_error
    return {
        "mean": mean,
        "standard_error": standard_error,
        "statistic_used": stat_name,
        "critical_value": critical,
        "margin_of_error": margin,
        "lower": mean - margin,
        "upper": mean + margin,
    }


def two_sample_ttest(group_a, group_b, alpha=0.05):
    """
    Two-sample (independent) t-test comparing the means of two groups.
    Welch's version (equal_var=False) because the groups can have
    different spreads. Degrees of freedom uses the conservative approach
    from the notes: the smaller group size minus 1.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)

    # conservative degrees of freedom = smaller sample - 1
    df_cons = min(len(a), len(b)) - 1

    if p_value < 0.001:
        p_text = "< 0.001"
    else:
        p_text = "%.4f" % p_value

    return {
        "n_a": int(len(a)),
        "mean_a": float(np.mean(a)),
        "std_a": float(np.std(a, ddof=1)),
        "n_b": int(len(b)),
        "mean_b": float(np.mean(b)),
        "std_b": float(np.std(b, ddof=1)),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "p_value_text": p_text,
        "df_conservative": df_cons,
        "reject_null": bool(p_value <= alpha),
    }
