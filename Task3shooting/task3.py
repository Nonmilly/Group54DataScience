# task3.py
# Author: Mildred
# Task 3 - SHOOTING EFFICIENCY
#
# This version reads the World Cup data from an Excel file (no downloading).
# The file is  data/fifadata.xlsx  and has these columns:
#     Team, Matches, Goals, Attempts
#
# Run it with:   python task3.py
#
# It covers the six required skills and follows the 4-Step Process for a
# hypothesis test from the unit notes (State / Plan / Solve / Conclude).

import os
import pandas as pd

from stats_helpers import (
    describe,
    check_normality,
    confidence_interval,
    two_sample_ttest,
)

# ---------------------------------------------------------------------
# WHERE THE DATA COMES FROM
# ---------------------------------------------------------------------
# The data lives in an Excel workbook that sits in the data/ folder next
# to this script. I open it with pandas.read_excel(). Nothing is
# downloaded - the file is prepared beforehand and read straight off disk.
DATA_FILE = os.path.join(os.path.dirname(__file__), "data", "fifadata.xlsx")

LINE = "=" * 70


def heading(text):
    """print a title with a line under it so the output is easy to read"""
    print()
    print(LINE)
    print(text)
    print(LINE)


def load_data():
    """
    Read the World Cup data from the Excel file into a DataFrame.
    Returns one row per team.
    """
    print("Reading data from:", DATA_FILE)

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            "could not find data/fifadata.xlsx - put the Excel file in the "
            "data folder next to this script"
        )

    # read_excel needs the openpyxl package to open .xlsx files
    df = pd.read_excel(DATA_FILE)
    print("  loaded", len(df), "teams from the workbook")
    return df


def main():
    heading("TASK 3 - SHOOTING EFFICIENCY (Mildred)")
    print("FIFA World Cup 2026 - how well teams turn attempts into goals")

    heading("GETTING THE DATA")
    try:
        df = load_data()
    except Exception as e:
        print("\nCould not read the Excel file:", e)
        return

    print("\nFirst 10 rows of the data:")
    print(df.head(10).to_string(index=False))

    # =================================================================
    # SKILL 1 - ANALYTIC QUESTION            (STEP 1: STATE)
    # =================================================================
    heading("STEP 1: STATE - the analytic question")
    print("How efficiently do teams convert their attempts at goal into")
    print("goals at the 2026 World Cup, and do high-volume shooting teams")
    print("convert at a different rate from low-volume shooting teams?")
    print()
    print("Response variable    : Conversion = Goals / Attempts")
    print("Explanatory variable : shooting volume (high or low)")
    print("Level of measurement : ratio, continuous")

    # =================================================================
    # STEP 2: PLAN - the hypotheses
    # =================================================================
    heading("STEP 2: PLAN - the hypotheses")
    print("H0: mu_high  = mu_low    both groups convert at the same rate")
    print("Ha: mu_high != mu_low    the two groups differ  (two-sided)")
    print()
    print("Test  : two-sample (independent) t-test")
    print("alpha : 0.05")

    # =================================================================
    # SKILL 2 - DATA WRANGLING               (STEP 3: SOLVE)
    # =================================================================
    heading("SKILL 2: DATA WRANGLING")
    rows_before = len(df)
    df = df[["Team", "Matches", "Goals", "Attempts"]].copy()
    for c in ["Matches", "Goals", "Attempts"]:
        df[c] = df[c].astype(float)

    # listwise (case) deletion - drop a row that is missing anything I need
    df = df.dropna(subset=["Matches", "Goals", "Attempts"])
    # conversion divides by Attempts, so a team with none would break it
    df = df[df["Attempts"] > 0]

    print("Columns kept        : Team, Matches, Goals, Attempts")
    print("Missing value method: listwise (case) deletion")
    print("Rows before / after : %d / %d  (%d dropped)"
          % (rows_before, len(df), rows_before - len(df)))

    # =================================================================
    # SKILL 3 - DATA PREPARATION AND SAMPLING
    # =================================================================
    heading("SKILL 3: DATA PREPARATION AND SAMPLING")
    # feature construction - build the variable I actually study
    df["Conversion"] = df["Goals"] / df["Attempts"]
    population_size = len(df)
    sample = df["Conversion"].sample(n=min(30, population_size), random_state=3)

    print("Feature construction: Conversion = Goals / Attempts")
    print("Population          : %d teams" % population_size)
    print("Sample              : %d teams" % len(sample))
    print("Sampling method     : simple random sampling (random_state=3)")
    print("Why 30              : the Central Limit Theorem needs n >= 30")

    print("\nThe five best converters in the whole population:")
    best = df.sort_values("Conversion", ascending=False).head(5)
    print(best[["Team", "Goals", "Attempts", "Conversion"]]
          .round(4).to_string(index=False))

    # =================================================================
    # CONDITIONS FOR INFERENCE               (STEP 3: SOLVE)
    # =================================================================
    heading("CONDITIONS FOR INFERENCE")
    conditions = check_normality(sample)
    print("1. Simple random sample : yes, .sample() draws at random")
    print("2. Roughly normal       : skewness %.3f -> %s"
          % (conditions["skewness"],
             "yes" if conditions["roughly_normal"] else "not clearly"))
    print("   (mean %.4f vs median %.4f - close together means symmetric)"
          % (conditions["mean"], conditions["median"]))

    # =================================================================
    # SKILL 4 - DESCRIPTIVE STATISTICS
    # =================================================================
    heading("SKILL 4: DESCRIPTIVE STATISTICS")
    d = describe(sample)
    print("Central tendency")
    print("   n                  %d" % d["n"])
    print("   mean               %.4f   (%.1f%% of attempts score)"
          % (d["mean"], d["mean"] * 100))
    print("   median             %.4f" % d["median"])
    print("Dispersion")
    print("   minimum            %.4f" % d["min"])
    print("   maximum            %.4f" % d["max"])
    print("   range              %.4f" % d["range"])
    print("   Q1 / Q3            %.4f / %.4f" % (d["q1"], d["q3"]))
    print("   IQR                %.4f" % d["iqr"])
    print("   variance           %.5f" % d["variance"])
    print("   standard deviation %.4f" % d["std"])

    # =================================================================
    # SKILL 5 - CONFIDENCE INTERVAL
    # =================================================================
    heading("SKILL 5: CONFIDENCE INTERVAL (95%)")
    ci = confidence_interval(sample, 0.95)
    print("Formula: CI = x-bar +/- z* . (s / sqrt(n))")
    print()
    print("   sample mean        %.4f" % ci["mean"])
    print("   standard error     %.5f" % ci["standard_error"])
    print("   statistic used     %s = %.4f   (n >= 30, so z* by the rule)"
          % (ci["statistic_used"], ci["critical_value"]))
    print("   margin of error    %.5f" % ci["margin_of_error"])
    print()
    print("   95%% CI = [%.4f , %.4f]" % (ci["lower"], ci["upper"]))
    print()
    print("   We are 95% confident the true mean conversion rate of all")
    print("   teams is between %.1f%% and %.1f%%."
          % (ci["lower"] * 100, ci["upper"] * 100))

    # =================================================================
    # SKILL 6 - TWO-SAMPLE t-TEST
    # =================================================================
    heading("SKILL 6: TWO-SAMPLE t-TEST")
    # discretisation - split a continuous variable into two categories.
    # attempts PER MATCH, so teams that played more games aren't favoured.
    df["AttemptsPerMatch"] = df["Attempts"] / df["Matches"]
    median_attempts = df["AttemptsPerMatch"].median()
    high = df[df["AttemptsPerMatch"] >= median_attempts]["Conversion"]
    low = df[df["AttemptsPerMatch"] < median_attempts]["Conversion"]

    print("Split on the median of attempts per match: %.1f" % median_attempts)
    print("(per match, not the total, so teams that played more matches")
    print(" don't automatically land in the high group)")
    print()

    t = two_sample_ttest(high, low, alpha=0.05)
    print("   Group A - high volume : n %2d   mean %.4f   s %.4f"
          % (t["n_a"], t["mean_a"], t["std_a"]))
    print("   Group B - low volume  : n %2d   mean %.4f   s %.4f"
          % (t["n_b"], t["mean_b"], t["std_b"]))
    print()
    print("   t statistic        %.4f" % t["t_stat"])
    print("   p-value            %s" % t["p_value_text"])
    print("   degrees of freedom %d   (smaller group - 1, the conservative"
          % t["df_conservative"])
    print("                          approach from the notes)")
    print("   Welch's test       equal_var=False, variances kept separate")

    # =================================================================
    # STEP 4: CONCLUDE
    # =================================================================
    heading("STEP 4: CONCLUDE")
    if t["p_value"] < 0.0001:
        chance = "fewer than 1 time in 10,000"
    else:
        chance = "about %.2f%% of the time" % (t["p_value"] * 100)

    print("The average team converts %.1f%% of its attempts at goal, and we"
          % (ci["mean"] * 100))
    print("are 95% confident the true average for all teams is between")
    print("%.1f%% and %.1f%%." % (ci["lower"] * 100, ci["upper"] * 100))
    print()
    if t["reject_null"]:
        print("The p-value is %s, at or below 0.05. A difference this big"
              % t["p_value_text"])
        print("between the groups would happen by chance %s," % chance)
        print("which is unusual.")
        print()
        print(">>> REJECT H0. There IS enough evidence that high-volume")
        print(">>> shooting teams convert at a different rate from")
        print(">>> low-volume shooting teams.")
    else:
        print("The p-value is %s, above 0.05. A difference this big between"
              % t["p_value_text"])
        print("the groups could still happen by chance %s," % chance)
        print("so it is not unusual.")
        print()
        print(">>> DO NOT REJECT H0. There is NOT enough evidence that")
        print(">>> high-volume shooting teams convert at a different rate")
        print(">>> from low-volume shooting teams.")
    print()
    print("Reported in APA format: t(%d) = %.2f, p = %s"
          % (t["df_conservative"], t["t_stat"],
             ("< .001" if t["p_value"] < 0.001
              else ("%.3f" % t["p_value"]).lstrip("0"))))
    print()
    print(LINE)
    print("End of Task 3.")
    print(LINE)


if __name__ == "__main__":
    main()
