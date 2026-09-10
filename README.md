# Replication Code: Deservingness and Inequality Aversion in the U.S. Federal Individual Income Tax System

This repository contains the replication code for the paper "Deservingness and Inequality Aversion in the U.S. Federal Individual Income Tax System."

## Requirements

- Python 3.12
- NumPy
- SciPy
- pandas

Install dependencies:

```bash
pip install -r requirements.txt
'''Data
The CPS ASEC data are publicly available from the U.S. Census Bureau:

Website: https://www.census.gov/programs-surveys/cps/data.html

Download the ASEC person-level files (pppub files)

Place them in the same directory as the Python scripts

Required Files
File	Tax Year	ASEC Survey Year
pppub19.csv	2018	2019
pppub20.csv	2019	2020
pppub21.csv	2020	2021
pppub22.csv	2021	2022
pppub23.csv	2022	2023
pppub24.csv	2023	2024
pppub25.csv	2024	2025
Required Variables
The code uses the following variables from the CPS ASEC pppub files:

FILESTAT: Filing status

TAX_ID: Tax unit identifier

AGI: Adjusted Gross Income

CAP_VAL: Net capital gains

DIV_VAL: Dividend income

TAX_INC: Taxable income (from Census tax model)

FEDTAX_AC: Federal tax after credits

FEDTAX_BC: Federal tax before credits

SPM_EITC: Earned Income Tax Credit

SPM_ACTC: Additional Child Tax Credit

MARSUPWT: March Supplement person weight

EIP_CRD: Economic Impact Payment (2020 and 2021 files only)

Code Structure
Baseline Estimation by Tax Year
File	Tax Year	Description
asec2019_baseline.py	2018	Baseline MFJ estimation
asec2020_baseline.py	2019	Baseline MFJ estimation
asec2021_baseline.py	2020	Baseline MFJ estimation, with and without EIP
asec2022_baseline.py	2021	Baseline MFJ estimation, with and without EIP
asec2023_baseline.py	2022	Baseline MFJ estimation
asec2024_baseline.py	2023	Baseline MFJ estimation
asec2025_baseline.py	2024	Baseline MFJ estimation
Robustness Checks (Tax Year 2023)
File	Description
asec2024_unweighted.py	All survey weights set to 1
asec2024_no_credits.py	EITC/ACTC excluded from disposable income
asec2024_pref_income.py	Alternative treatments of preferential income
asec2024_single.py	Single filing status, with and without EITC/ACTC
asec2024_hh.py	Head of Household filing status, with and without EITC/ACTC
Running the Code
Each file is self-contained. Run any file with:

bash
python asec2024_baseline.py
What Each Script Does
Loads the specified CPS ASEC data file

Constructs the analysis sample (filing status, positive AGI)

Constructs tax variables (ordinary taxable income, preferential tax, NIIT)

Runs the joint grid search over g and σ

Reports the best-fitting parameters for:

Population-weighted WAAD

Tax-weighted WAAD

Prints detailed results for manual checking

Key Settings
Each script has a settings section at the top where you can adjust:

USE_WEIGHTS: True for weighted, False for unweighted

LTCG_SHARE: Share of net capital gains treated as long-term

QDIV_SHARE: Share of dividends treated as qualified

G_GRID: Grid values for the government revenue share

SIGMA_GRID: Grid values for the inequality aversion parameter

Output
The scripts print:

Sample size and descriptive statistics

Observed government revenue share

Bracket population weights

Detailed table of optimal rates for selected σ values

Joint search results for each g


Global best results for population-weighted and tax-weighted criteria

Notes on Numerical Precision
The objective function is flat near the optimum. The exact numerical minimum may vary slightly across computing environments or repeated runs. The substantive conclusions are based on the broad pattern of the WAAD surface, which is robust across environments.

License
MIT License

text

