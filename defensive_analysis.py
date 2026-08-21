import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load the Excel dataset
df = pd.read_excel("defensive_stats.xlsx")

# Show first 5 rows
print(df.head())

# Show number of rows and columns
print("Dataset shape:", df.shape)

# Show column names
print("Columns:")
print(df.columns.tolist())

# Keep only the columns needed for our analysis
def_data = df[
    ["Player", "Pos", "Squad", "90s", "TklW", "Int"]
].copy()

print("\nSelected data:")
print(def_data.head())

print("\nPosition counts:")
print(def_data["Pos"].value_counts())

# Keep only pure defenders and midfielders
def_data = def_data[
    def_data["Pos"].isin(["DF", "MF"])
].copy()

print("\nAfter filtering positions:")
print(def_data["Pos"].value_counts())


# Keep only players who played at least 1 full 90-minute equivalent
def_data = def_data[
    def_data["90s"] >= 1
].copy()

print("\nAfter removing players with less than 1.0 90s:")
print(def_data["Pos"].value_counts())

print("\nRemaining rows:")
print(def_data.shape)

print("\nMissing values:")
print(def_data.isnull().sum())

def_data["DefActions90"] = (
    def_data["TklW"] + def_data["Int"]
) / def_data["90s"]

print("\nDefensive actions per 90:")
print(
    def_data[
        ["Player", "Pos", "90s", "TklW", "Int", "DefActions90"]
    ].head(10)
)

# Check missing values
print("\nMissing values:")
print(def_data.isnull().sum())


# Calculate defensive actions per 90
def_data["DefActions90"] = (
    def_data["TklW"] + def_data["Int"]
) / def_data["90s"]


# Display calculated values
print("\nDefensive actions per 90:")
print(
    def_data[
        ["Player", "Pos", "90s", "TklW", "Int", "DefActions90"]
    ].head(10)
)

# Separate defenders and midfielders
defenders = def_data[def_data["Pos"] == "DF"]
midfielders = def_data[def_data["Pos"] == "MF"]

# Randomly sample 30 players from each group
def_sample = defenders.sample(n=30, random_state=42)
mf_sample = midfielders.sample(n=30, random_state=42)

# Combine both samples
sample = pd.concat([def_sample, mf_sample])

print("\nSample size:")
print(sample["Pos"].value_counts())

print("\nSample preview:")
print(sample.head())

# Descriptive statistics for defensive actions per 90
print("\nOverall descriptive statistics:")
print(sample["DefActions90"].describe())

print("\nDescriptive statistics by position:")
print(
    sample.groupby("Pos")["DefActions90"].describe()
)

# Separate the defensive actions values for each group
def_values = def_sample["DefActions90"]
mf_values = mf_sample["DefActions90"]

# 95% confidence interval for defenders
def_mean = def_values.mean()
def_ci = stats.t.interval(
    confidence=0.95,
    df=len(def_values) - 1,
    loc=def_mean,
    scale=stats.sem(def_values)
)

# 95% confidence interval for midfielders
mf_mean = mf_values.mean()
mf_ci = stats.t.interval(
    confidence=0.95,
    df=len(mf_values) - 1,
    loc=mf_mean,
    scale=stats.sem(mf_values)
)

print("\n95% Confidence Intervals:")

print(
    "Defenders:",
    round(def_mean, 3),
    "CI =",
    (round(def_ci[0], 3), round(def_ci[1], 3))
)

print(
    "Midfielders:",
    round(mf_mean, 3),
    "CI =",
    (round(mf_ci[0], 3), round(mf_ci[1], 3))
)


# Welch two-sample t-test
t_stat, p_value = stats.ttest_ind(
    def_values,
    mf_values,
    equal_var=False
)

print("\nWelch Two-Sample T-Test:")
print("T-statistic:", round(t_stat, 4))
print("P-value:", round(p_value, 4))

# Decision at 5% significance level
alpha = 0.05

if p_value < alpha:
    print("Result: Reject the null hypothesis.")
    print("There is a statistically significant difference between defenders and midfielders.")
else:
    print("Result: Fail to reject the null hypothesis.")
    print("There is not enough evidence of a statistically significant difference between defenders and midfielders.")


    # Boxplot comparing defenders and midfielders
sample.boxplot(
    column="DefActions90",
    by="Pos"
)

plt.xlabel("Position")
plt.ylabel("Tackles Won + Interceptions per 90")
plt.title("Tackles Won + Interceptions per 90 by Position")
plt.suptitle("")

plt.savefig(
    "defensive_actions_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Histogram of defensive actions per 90
plt.hist(
    def_sample["DefActions90"],
    bins=10,
    alpha=0.6,
    label="Defenders"
)

plt.hist(
    mf_sample["DefActions90"],
    bins=10,
    alpha=0.6,
    label="Midfielders"
)

plt.xlabel("Tackles Won + Interceptions per 90")
plt.ylabel("Number of Players")
plt.title("Distribution of Tackles Won + Interceptions per 90")
plt.legend()

plt.savefig(
    "defensive_actions_histogram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()