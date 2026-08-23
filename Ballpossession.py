''' 
=====================================================================================
                                    Assignment 2:
=====================================================================================
Analytic Question; 
------------------
Did the match winners in the FIFA 2026 World Cup matches have more ball possession 
than the losing team?

Key Tasks; 
----------
1. Data Wrangling
2. Data preparation and sampling
3. Descriptive statistics
4. Inferential statistics (Confidence interval)
5. Inferential statistics (One-Sample t-Test)
======================================================================================
'''
#Calling the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Loading the CSV dataset
df = pd.read_csv("FWC2026_PossessionData.csv")

# Showing the first 5 rows of the dataset, its shape, and column names
print("First 5 rows:")
print(df.head())
print("Dataset shape:", df.shape)
print("Columns:")
print(df.columns.tolist())

# Keeping only the columns needed for our analysis
poss_data = df[
    ["Match_No", "Stage", "Home_Team (HT)", "Away_Team",
     "Winner", "HT_Possession (%)", "AT_Possession (%)"]
].copy()

print("\nOverview of the data:")
print(poss_data.head())

print("\nWinner value counts:")
print(poss_data["Winner"].value_counts().head())

#1. Data Wrangling
#=================
''' 
Drop drawn matches.
A draw has no winner/loser, so it can't be considered for our analysis. 
Only keep matches with a winner and a loser = Population (N).
Checking for missing values in the dataset
'''
poss_data = poss_data[
    poss_data["Winner"] != "Draw"
].copy()

print("\nMatches with winners and losers:")
print("Population (N):", poss_data.shape[0])

print("\nMissing values:")
print(poss_data.isnull().sum())

'''
Assigning ball possession to the match winners and losers 
regardless of if they are the home or away team.
Defining the population size and the number of drawn matches
'''
def split_possession(row):
    if row["Winner"] == row["Home_Team (HT)"]:
        return pd.Series([row["HT_Possession (%)"], row["AT_Possession (%)"]])
    else:
        return pd.Series([row["AT_Possession (%)"], row["HT_Possession (%)"]])

poss_data[["Winner_Poss", "Loser_Poss"]] = poss_data.apply(split_possession, axis=1)

# Poss_gap = possession gap per match -> Winner possession - Loser possession
# Positive means winner had more possession
# Negative means loser had more possession
poss_data["Poss_gap"] = poss_data["Winner_Poss"] - poss_data["Loser_Poss"]

print("\n***Possession Gap***")
print(
    poss_data[
        ["Match_No", "Winner", "Winner_Poss", "Loser_Poss", "Poss_gap"]
    ].head(10)
)
#2. Data preparation and sampling
#================================
# Simple random sample, n=40 (>=30 satisfies CLT), random_state fixed for reproducibility
poss_sample = poss_data.sample(n=40, random_state=42)

print("\nSample size:")
print(poss_sample.shape[0])

print("\nSample preview:")
print(poss_sample.head())

#3. Descriptive statistics
#=========================
# Overall Descriptive statistics

print("\n***Overall Descriptive Statistics***")
print(poss_sample[["Winner_Poss", "Loser_Poss", "Poss_gap"]].describe())

# Normality check
# Justification for using a t-test in the section below.
print("\n****Normality Check (Shapiro-Wilk Test)****")
w_stat, w_p = stats.shapiro(poss_sample["Poss_gap"])
print("\nShapiro-Wilk normality test:")
print("W:", round(w_stat, 4), "p:", round(w_p, 4))

# 4. Inferential statistics (Confidence interval)
#================================================
# 95% confidence interval for the mean possession gap
Possgap_values = poss_sample["Poss_gap"]
Possgap_mean = Possgap_values.mean()
Possgap_ci = stats.t.interval(
    confidence=0.95,
    df=len(Possgap_values) - 1,
    loc=Possgap_mean,
    scale=stats.sem(Possgap_values)
)

print("\n95% Confidence Interval:")
print(
    "Possession gap:",
    round(Possgap_mean, 3),
    "CI =",
    (round(Possgap_ci[0], 3), round(Possgap_ci[1], 3))
)
# 5. Inferential statistics (One-Sample t-Test)
#==============================================
'''
One-sample t-test: H0: possgap mean = 0, H1: mean gap > 0 
One-tailed
t-test chosen because Central Limit Theorem holds
CLT conditions met; sample size >= 30, normality check passed (p > 0.05)
'''
t_stat, p_value = stats.ttest_1samp(
    Possgap_values,
    popmean=0
)

# Returns a two-tailed p-value; halve it since H1 is one-directional
p_value_one_tailed = p_value / 2 if t_stat > 0 else 1 - p_value / 2

print("\nOne-Sample T-Test:")
print("T-statistic:", round(t_stat, 4))
print("P-value (one-tailed):", round(p_value_one_tailed, 6))

# Decision at 5% significance level
alpha = 0.05

if p_value_one_tailed < alpha:
    print("Result: Reject the null hypothesis.")
    print("Winners have significantly higher possession than losers, on average.")
else:
    print("Result: Fail to reject the null hypothesis.")
    print("There is not enough evidence that winners have higher possession, on average.")

# 6. Visualizations
#==================
# Boxplot comparing Winner vs Loser possession
poss_sample[["Winner_Poss", "Loser_Poss"]].boxplot()

plt.ylabel("Ball Possession (%)")
plt.title("Ball Possession Box Plot: Winner vs Loser")

plt.savefig(
    "possession_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Histogram of possession gap
plt.hist(
    poss_sample["Poss_gap"],
    bins=10,
    alpha=0.7,
    color="orange",
    edgecolor="black"
)

plt.axvline(0, color="Blue", linestyle="-", label="No possession advantage")
plt.xlabel("Possession Gap (Winner - Loser, %)")
plt.ylabel("Number of Matches")
plt.title("Distribution of Possession Gap")
plt.legend()

plt.savefig(
    "possession_gap_histogram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
