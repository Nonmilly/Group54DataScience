# ============================================================
# Task 2 - Passing Performance Analysis
# FIFA World Cup 2026
# ============================================================

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------

TEAM_FILE = os.path.join(
    "data",
    "FIFA_World_Cup_2026_Team_Distribution.xlsx"
)

STANDINGS_FILE = os.path.join(
    "data",
    "FIFA_World_Cup_2026_Cleaned_Standings.xlsx"
)

OUTPUT_DIR = "task2_passing_performance"

FIGURES_DIR = os.path.join(
    OUTPUT_DIR,
    "figures"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "passing_performance_analysis.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. Load datasets
# ------------------------------------------------------------

team_data = pd.read_excel(TEAM_FILE)
standings_data = pd.read_excel(STANDINGS_FILE)

print("\n============================================================")
print("DATA LOADED")
print("============================================================")

print(f"Team distribution rows: {len(team_data)}")
print(f"Standings rows: {len(standings_data)}")


# ------------------------------------------------------------
# 3. Clean column names
# ------------------------------------------------------------

team_data.columns = team_data.columns.str.strip()
standings_data.columns = standings_data.columns.str.strip()


# ------------------------------------------------------------
# 4. Find required columns
# ------------------------------------------------------------

def find_column(df, possible_names):

    for name in possible_names:
        if name in df.columns:
            return name

    lower_columns = {
        col.lower(): col
        for col in df.columns
    }

    for name in possible_names:
        if name.lower() in lower_columns:
            return lower_columns[name.lower()]

    return None


team_column = find_column(
    team_data,
    [
        "Team",
        "team",
        "Country",
        "country"
    ]
)

passing_accuracy_column = find_column(
    team_data,
    [
        "Passing Accuracy (%)",
        "Passing Accuracy",
        "passing accuracy (%)"
    ]
)

standings_team_column = find_column(
    standings_data,
    [
        "Team",
        "team",
        "Country",
        "country"
    ]
)

progression_column = find_column(
    standings_data,
    [
        "Progression",
        "progression",
        "Status",
        "status"
    ]
)


# ------------------------------------------------------------
# 5. Validate required columns
# ------------------------------------------------------------

if team_column is None:
    raise ValueError(
        "Could not find the Team column in the team distribution dataset."
    )

if passing_accuracy_column is None:
    raise ValueError(
        "Could not find the Passing Accuracy (%) column."
    )

if standings_team_column is None:
    raise ValueError(
        "Could not find the Team column in the standings dataset."
    )

if progression_column is None:
    raise ValueError(
        "Could not find the Progression column in the standings dataset."
    )


print("\n============================================================")
print("REQUIRED COLUMNS")
print("============================================================")

print(f"Team column: {team_column}")
print(f"Passing Accuracy column: {passing_accuracy_column}")
print(f"Standings Team column: {standings_team_column}")
print(f"Progression column: {progression_column}")


# ------------------------------------------------------------
# 6. Select required variables
# ------------------------------------------------------------

passing = team_data[
    [
        team_column,
        passing_accuracy_column
    ]
].copy()

standings = standings_data[
    [
        standings_team_column,
        progression_column
    ]
].copy()


# Rename columns to standard names

passing = passing.rename(
    columns={
        team_column: "Team",
        passing_accuracy_column: "Passing Accuracy (%)"
    }
)

standings = standings.rename(
    columns={
        standings_team_column: "Team",
        progression_column: "Progression"
    }
)


# ------------------------------------------------------------
# 7. Clean passing dataset
# ------------------------------------------------------------

passing["Team"] = (
    passing["Team"]
    .astype(str)
    .str.strip()
)

passing["Passing Accuracy (%)"] = pd.to_numeric(
    passing["Passing Accuracy (%)"],
    errors="coerce"
)

passing = passing.dropna(
    subset=[
        "Team",
        "Passing Accuracy (%)"
    ]
)


# ------------------------------------------------------------
# 8. FIFA team-code conversion
# ------------------------------------------------------------

print("\n============================================================")
print("TEAM CODE CONVERSION")
print("============================================================")

print("Converting FIFA team codes to team names...")


fifa_code_to_team = {

    "MEX": "Mexico",
    "RSA": "South Africa",
    "KOR": "Korea Republic",
    "CZE": "Czechia",
    "SUI": "Switzerland",
    "CAN": "Canada",
    "BIH": "Bosnia and Herzegovina",
    "QAT": "Qatar",
    "BRA": "Brazil",
    "MAR": "Morocco",
    "SCO": "Scotland",
    "HAI": "Haiti",
    "USA": "United States",
    "AUS": "Australia",
    "PAR": "Paraguay",
    "TUR": "Türkiye",
    "GER": "Germany",
    "CIV": "Côte d'Ivoire",
    "ECU": "Ecuador",
    "CUW": "Curaçao",
    "NED": "Netherlands",
    "JPN": "Japan",
    "SWE": "Sweden",
    "TUN": "Tunisia",
    "BEL": "Belgium",
    "EGY": "Egypt",
    "IRN": "IR Iran",
    "NZL": "New Zealand",
    "ESP": "Spain",
    "CPV": "Cabo Verde",
    "URU": "Uruguay",
    "KSA": "Saudi Arabia",
    "FRA": "France",
    "NOR": "Norway",
    "SEN": "Senegal",
    "IRQ": "Iraq",
    "ARG": "Argentina",
    "AUT": "Austria",
    "ALG": "Algeria",
    "JOR": "Jordan",
    "COL": "Colombia",
    "POR": "Portugal",
    "COD": "Congo DR",
    "UZB": "Uzbekistan",
    "ENG": "England",
    "CRO": "Croatia",
    "GHA": "Ghana",
    "PAN": "Panama"
}


# ------------------------------------------------------------
# 9. Clean standings team names
# ------------------------------------------------------------

standings["Team"] = (
    standings["Team"]
    .astype(str)
    .str.strip()
)

standings["Team"] = standings["Team"].replace(
    fifa_code_to_team
)


# ------------------------------------------------------------
# 10. Standardise team-name differences
# ------------------------------------------------------------

# Both datasets must use exactly the same team names.
#
# The passing dataset contains some names that differ from
# the names in the standings dataset. These aliases ensure
# that all 48 teams can be matched correctly.

team_name_aliases = {

    # United States
    "USA": "United States",
    "United States of America": "United States",

    # Panama
    "PAN": "Panama",

    # Cabo Verde
    "Cape Verde": "Cabo Verde",
    "Cabo Verde": "Cabo Verde",

    # Democratic Republic of the Congo
    "DR Congo": "Congo DR",
    "Congo DR": "Congo DR",

    # Türkiye
    "Turkey": "Türkiye",
    "Türkiye": "Türkiye",

    # Côte d'Ivoire
    "Ivory Coast": "Côte d'Ivoire",
    "Cote d'Ivoire": "Côte d'Ivoire",

    # Iran
    "Iran": "IR Iran",
    "IR Iran": "IR Iran",

    # Korea
    "South Korea": "Korea Republic",
    "Korea Republic": "Korea Republic"
}


passing["Team"] = (
    passing["Team"]
    .replace(team_name_aliases)
    .astype(str)
    .str.strip()
)

standings["Team"] = (
    standings["Team"]
    .replace(team_name_aliases)
    .astype(str)
    .str.strip()
)


# ------------------------------------------------------------
# 11. Clean progression labels
# ------------------------------------------------------------

def clean_progression(value):

    value = str(value).strip().lower()

    value = value.replace("-", " ")
    value = value.replace("_", " ")

    value = " ".join(value.split())

    return value


standings["Progression"] = (
    standings["Progression"]
    .apply(clean_progression)
)

standings = standings.dropna(
    subset=[
        "Team",
        "Progression"
    ]
)


# ------------------------------------------------------------
# 12. Display converted standings
# ------------------------------------------------------------

print("\nConverted standings teams:")

print(
    standings[
        [
            "Team",
            "Progression"
        ]
    ].head(10)
)


# ------------------------------------------------------------
# 13. Check dataset sizes before merge
# ------------------------------------------------------------

print("\n============================================================")
print("DATASET CHECK")
print("============================================================")

print(f"Passing dataset teams: {len(passing)}")
print(f"Standings dataset teams: {len(standings)}")


# ------------------------------------------------------------
# 14. Check unmatched teams BEFORE merge
# ------------------------------------------------------------

passing_teams = set(
    passing["Team"]
)

standings_teams = set(
    standings["Team"]
)

unmatched_passing = sorted(
    passing_teams - standings_teams
)

unmatched_standings = sorted(
    standings_teams - passing_teams
)

print("\nTeams in passing dataset but not standings:")
print(unmatched_passing)

print("\nTeams in standings but not passing dataset:")
print(unmatched_standings)


# ------------------------------------------------------------
# 15. Validate all teams matched
# ------------------------------------------------------------

if unmatched_passing or unmatched_standings:

    raise ValueError(
        "\nTeam-name matching is incomplete.\n"
        f"Unmatched passing teams: {unmatched_passing}\n"
        f"Unmatched standings teams: {unmatched_standings}\n"
        "Please check the team-name standardisation."
    )


# ------------------------------------------------------------
# 16. Merge datasets
# ------------------------------------------------------------

data = pd.merge(
    passing,
    standings,
    on="Team",
    how="inner",
    validate="one_to_one"
)


print("\n============================================================")
print("MERGED DATASET")
print("============================================================")

print(
    f"Number of teams after merging: {len(data)}"
)


# ------------------------------------------------------------
# 17. Validate merged dataset
# ------------------------------------------------------------

if len(data) != 48:

    raise ValueError(
        f"Expected 48 teams after merging, "
        f"but found {len(data)}."
    )


print("\nFirst five observations:")
print(data.head())


# ------------------------------------------------------------
# 18. Check progression groups
# ------------------------------------------------------------

print("\n============================================================")
print("PROGRESSION GROUPS")
print("============================================================")

progression_counts = (
    data["Progression"]
    .value_counts()
)

print(progression_counts)


# Validate expected groups

if "progressed" not in progression_counts:
    raise ValueError(
        "The 'progressed' group was not found."
    )

if "group stage eliminated" not in progression_counts:
    raise ValueError(
        "The 'group stage eliminated' group was not found."
    )


if progression_counts["progressed"] != 32:
    raise ValueError(
        "Expected 32 progressed teams, "
        f"but found {progression_counts['progressed']}."
    )


if progression_counts["group stage eliminated"] != 16:
    raise ValueError(
        "Expected 16 group-stage eliminated teams, "
        f"but found {progression_counts['group stage eliminated']}."
    )


# ------------------------------------------------------------
# 19. Descriptive statistics
# ------------------------------------------------------------

print("\n============================================================")
print("DESCRIPTIVE STATISTICS")
print("============================================================")

overall = data["Passing Accuracy (%)"]

print("\nOverall Passing Accuracy:")

print(
    f"Sample size (n): {overall.count()}"
)

print(
    f"Mean: {overall.mean():.2f}%"
)

print(
    f"Median: {overall.median():.2f}%"
)

print(
    f"Standard deviation: "
    f"{overall.std(ddof=1):.2f}"
)

print(
    f"Minimum: {overall.min():.2f}%"
)

print(
    f"Maximum: {overall.max():.2f}%"
)


print("\nDescriptive statistics by progression:")

group_descriptive = (
    data
    .groupby("Progression")[
        "Passing Accuracy (%)"
    ]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max"
        ]
    )
    .round(2)
)

print(group_descriptive)


# ------------------------------------------------------------
# 20. 95% Confidence Interval for overall mean
# ------------------------------------------------------------

n = overall.count()
mean = overall.mean()
std = overall.std(ddof=1)

if n < 2:
    raise ValueError(
        "At least two observations are required "
        "to calculate the confidence interval."
    )

standard_error = (
    std / np.sqrt(n)
)

t_critical = stats.t.ppf(
    0.975,
    df=n - 1
)

margin_of_error = (
    t_critical *
    standard_error
)

ci_lower = (
    mean -
    margin_of_error
)

ci_upper = (
    mean +
    margin_of_error
)


print("\n============================================================")
print("95% CONFIDENCE INTERVAL")
print("============================================================")

print(
    f"Sample mean: {mean:.2f}%"
)

print(
    f"95% CI lower bound: {ci_lower:.2f}%"
)

print(
    f"95% CI upper bound: {ci_upper:.2f}%"
)

print(
    f"95% CI: "
    f"({ci_lower:.2f}%, {ci_upper:.2f}%)"
)


# ------------------------------------------------------------
# 21. Prepare two progression groups
# ------------------------------------------------------------

progressed = data.loc[
    data["Progression"] == "progressed",
    "Passing Accuracy (%)"
].dropna()

eliminated = data.loc[
    data["Progression"] == "group stage eliminated",
    "Passing Accuracy (%)"
].dropna()


print("\n============================================================")
print("TWO GROUPS")
print("============================================================")

print(
    f"Progressed teams: {len(progressed)}"
)

print(
    f"Group-stage eliminated teams: "
    f"{len(eliminated)}"
)

print(
    f"Progressed mean: "
    f"{progressed.mean():.2f}%"
)

print(
    f"Group-stage eliminated mean: "
    f"{eliminated.mean():.2f}%"
)


# ------------------------------------------------------------
# 22. Validate groups
# ------------------------------------------------------------

if len(progressed) < 2:

    raise ValueError(
        "The Progressed group contains fewer "
        "than two observations."
    )


if len(eliminated) < 2:

    raise ValueError(
        "The Group Stage Eliminated group contains "
        "fewer than two observations."
    )


# ------------------------------------------------------------
# 23. Two-sample Welch t-test
# ------------------------------------------------------------

t_statistic, p_value = stats.ttest_ind(
    progressed,
    eliminated,
    equal_var=False,
    nan_policy="omit"
)


print("\n============================================================")
print("TWO-SAMPLE T-TEST")
print("============================================================")

print(
    f"t-statistic: {t_statistic:.4f}"
)

print(
    f"p-value: {p_value:.4f}"
)


# ------------------------------------------------------------
# 24. Hypothesis decision
# ------------------------------------------------------------

alpha = 0.05

if p_value < alpha:

    decision = (
        "Reject the null hypothesis "
        "at the 5% significance level."
    )

else:

    decision = (
        "Fail to reject the null hypothesis "
        "at the 5% significance level."
    )


print(
    f"Decision: {decision}"
)


# ------------------------------------------------------------
# 25. 95% CI for difference between group means
# ------------------------------------------------------------

progressed_variance = (
    progressed.var(ddof=1)
)

eliminated_variance = (
    eliminated.var(ddof=1)
)

mean_difference = (
    progressed.mean()
    -
    eliminated.mean()
)


se_difference = np.sqrt(
    (
        progressed_variance /
        len(progressed)
    )
    +
    (
        eliminated_variance /
        len(eliminated)
    )
)


# ------------------------------------------------------------
# Welch-Satterthwaite degrees of freedom
# ------------------------------------------------------------

numerator = (
    (
        progressed_variance /
        len(progressed)
    )
    +
    (
        eliminated_variance /
        len(eliminated)
    )
) ** 2


denominator = (
    (
        (
            progressed_variance /
            len(progressed)
        ) ** 2
        /
        (len(progressed) - 1)
    )
    +
    (
        (
            eliminated_variance /
            len(eliminated)
        ) ** 2
        /
        (len(eliminated) - 1)
    )
)


df_welch = (
    numerator /
    denominator
)


t_critical_difference = stats.t.ppf(
    0.975,
    df=df_welch
)


difference_margin = (
    t_critical_difference *
    se_difference
)


difference_ci_lower = (
    mean_difference -
    difference_margin
)

difference_ci_upper = (
    mean_difference +
    difference_margin
)


print("\n============================================================")
print("95% CI FOR DIFFERENCE IN GROUP MEANS")
print("============================================================")

print(
    "Mean difference "
    "(Progressed - Eliminated): "
    f"{mean_difference:.2f} percentage points"
)

print(
    f"Welch degrees of freedom: "
    f"{df_welch:.2f}"
)

print(
    f"95% CI: "
    f"({difference_ci_lower:.2f}, "
    f"{difference_ci_upper:.2f})"
)


# ------------------------------------------------------------
# 26. Create boxplot
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

data.boxplot(
    column="Passing Accuracy (%)",
    by="Progression"
)

plt.title(
    "Passing Accuracy by Tournament Progression"
)

plt.suptitle("")

plt.xlabel(
    "Tournament Progression"
)

plt.ylabel(
    "Passing Accuracy (%)"
)

plt.xticks(rotation=0)

plt.tight_layout()


boxplot_path = os.path.join(
    FIGURES_DIR,
    "passing_accuracy_by_progression.png"
)

plt.savefig(
    boxplot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 27. Create mean comparison chart
# ------------------------------------------------------------

group_means = (
    data
    .groupby("Progression")[
        "Passing Accuracy (%)"
    ]
    .mean()
    .sort_values(
        ascending=False
    )
)


plt.figure(figsize=(8, 6))

group_means.plot(
    kind="bar"
)

plt.title(
    "Average Passing Accuracy by Tournament Progression"
)

plt.xlabel(
    "Tournament Progression"
)

plt.ylabel(
    "Mean Passing Accuracy (%)"
)

plt.xticks(rotation=0)

plt.tight_layout()


mean_chart_path = os.path.join(
    FIGURES_DIR,
    "mean_passing_accuracy_by_progression.png"
)

plt.savefig(
    mean_chart_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 28. Create distribution histogram
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    progressed,
    bins=8,
    alpha=0.7,
    label="Progressed"
)

plt.hist(
    eliminated,
    bins=8,
    alpha=0.7,
    label="Group-stage eliminated"
)

plt.title(
    "Distribution of Passing Accuracy in the Sample"
)

plt.xlabel(
    "Passing Accuracy (%)"
)

plt.ylabel(
    "Number of teams"
)

plt.legend()

plt.tight_layout()


distribution_path = os.path.join(
    FIGURES_DIR,
    "passing_accuracy_distribution.png"
)

plt.savefig(
    distribution_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 29. Save analysis dataset
# ------------------------------------------------------------

data.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# 30. Final summary
# ------------------------------------------------------------

print("\n============================================================")
print("ANALYSIS COMPLETE")
print("============================================================")

print(
    f"Analysis dataset saved to: "
    f"{OUTPUT_FILE}"
)

print(
    f"Boxplot saved to: "
    f"{boxplot_path}"
)

print(
    f"Mean comparison chart saved to: "
    f"{mean_chart_path}"
)

print(
    f"Distribution histogram saved to: "
    f"{distribution_path}"
)

print(
    "\nTask 2 analysis completed successfully."
)