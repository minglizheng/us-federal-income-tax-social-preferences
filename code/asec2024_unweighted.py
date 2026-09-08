
"""
Robustness check: Unweighted estimation for tax year 2023
Married Filing Jointly, EITC and ACTC included as endogenous transfers
All survey weights set to 1

This file is identical to asec2024_baseline.py except USE_WEIGHTS = False.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings

# Suppress the delta_grad warning (flat objective near optimum)
warnings.filterwarnings('ignore', message='delta_grad == 0.0')

# =============================================================================
# SETTINGS
# =============================================================================

DATA_FILE = "pppub24.csv"           # 2024 ASEC, tax year 2023
FILING_STATUS_MAX = 4               # FILESTAT < 4 for married filing jointly

STD_DEDUCTION = 27700                # 2023 MFJ standard deduction
BRACKETS = np.array([22000, 89450, 190750, 364200, 462500, 693750])

# Preferential income calibration
LTCG_SHARE = 1.0                     # share of positive net capital gains treated as long-term
QDIV_SHARE = 0.75                    # share of dividends treated as qualified

# Grid search parameters
# G_GRID = [0.12, 0.13, 0.14, 0.15, 0.16]
# SIGMA_GRID = np.arange(1.00, 1.31, 0.01)



G_GRID = np.arange(0.125, 0.148, 0.001)
SIGMA_GRID = np.arange(1.108, 1.150, 0.001)

# Use survey weights? Set False for equal weights (this robustness check)
USE_WEIGHTS = False

# Detailed output for a specific g
PRINT_DETAILED_G = 0.145
DETAILED_SIGMA_VALUES = [1.000, 1.050, 1.080, 1.100, 1.103, 1.106, 1.107, 1.108, 1.110, 1.120, 1.130, 1.150, 1.200, 1.300]

# Solver settings
SOLVER_OPTIONS = {
    'gtol': 1e-8,
    'xtol': 1e-8,
    'barrier_tol': 1e-8,
    'maxiter': 1000,
    'verbose': 0
}

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================

print(f"Reading {DATA_FILE}...")

required_cols = [
    'FILESTAT', 'TAX_ID', 'AGI', 'CAP_VAL', 'DIV_VAL', 'TAX_INC',
    'FEDTAX_AC', 'FEDTAX_BC', 'SPM_EITC', 'SPM_ACTC',
    'SPM_CAPHOUSESUB', 'SPM_SNAPSUB', 'MARSUPWT'
]

df_full = pd.read_csv(DATA_FILE)
missing_cols = [col for col in required_cols if col not in df_full.columns]
if missing_cols:
    print(f"Warning: Missing columns: {missing_cols}")

available_cols = [col for col in required_cols if col in df_full.columns]
data = df_full[available_cols].copy()

# Select married filing jointly
df0 = data[data['FILESTAT'] < FILING_STATUS_MAX].copy()

# Identify primary filer and aggregate spouse information
df0 = df0.sort_values(['TAX_ID', 'AGI'], ascending=[True, False])
df0['is_filer'] = df0.groupby('TAX_ID')['AGI'].transform(
    lambda x: (x == x.max()) & (x >= 0)
)
df0['total_cap'] = df0.groupby('TAX_ID')['CAP_VAL'].transform('sum')
df0['total_div'] = df0.groupby('TAX_ID')['DIV_VAL'].transform('sum')
df0['filer_cap'] = df0['total_cap'] * df0['is_filer']
df0['filer_div'] = df0['total_div'] * df0['is_filer']

filer_data = df0[df0['is_filer']].copy()
filer_data = filer_data[filer_data['AGI'] > 0].copy()
df = filer_data

# =============================================================================
# SAMPLE DESCRIPTION
# =============================================================================

n_total = len(df)
total_weighted = df['MARSUPWT'].sum()
pos_weighted_cap = df.loc[df['CAP_VAL'] > 0, 'MARSUPWT'].sum()
pos_weighted_div = df.loc[df['DIV_VAL'] > 0, 'MARSUPWT'].sum()
pos_weighted_eitc = df.loc[df['SPM_EITC'] > 0, 'MARSUPWT'].sum()
pos_weighted_actc = df.loc[df['SPM_ACTC'] > 0, 'MARSUPWT'].sum()

print(f"\nSample size: {n_total}")
print(f"\nWeighted proportions (using survey weights):")
print(f"  CAP_VAL > 0: {pos_weighted_cap/total_weighted*100:.2f}%")
print(f"  DIV_VAL > 0: {pos_weighted_div/total_weighted*100:.2f}%")
print(f"  EITC  > 0:   {pos_weighted_eitc/total_weighted*100:.2f}%")
print(f"  ACTC  > 0:   {pos_weighted_actc/total_weighted*100:.2f}%")

# =============================================================================
# CONSTRUCT TAX VARIABLES
# =============================================================================

# Preferential income
df['ltcg_estimate'] = df['CAP_VAL'] * LTCG_SHARE
df['qualified_div_estimate'] = df['DIV_VAL'] * QDIV_SHARE
df['pref_income'] = df['ltcg_estimate'] + df['qualified_div_estimate']

# Ordinary taxable income
df['ordinary_income'] = (df['AGI'] - df['pref_income']).clip(lower=0)
df['taxable_ordinary'] = (df['ordinary_income'] - STD_DEDUCTION).clip(lower=0)

# Preferential tax
def calculate_preferential_tax(pref_income, agi):
    tax = np.zeros_like(pref_income)
    mask_15 = (agi > 89250) & (agi <= 553850)
    mask_20 = agi > 553850
    tax[mask_15] = pref_income[mask_15] * 0.15
    tax[mask_20] = pref_income[mask_20] * 0.20
    return tax

df['pref_tax'] = calculate_preferential_tax(df['pref_income'].values, df['AGI'].values)

# Net Investment Income Tax
df['niit_base'] = np.minimum(df['pref_income'], np.maximum(df['AGI'] - 250000, 0))
df['niit'] = np.where(df['AGI'] > 250000, df['niit_base'] * 0.038, 0)

# Extract arrays
agi = df['AGI'].values
taxable_ordinary = df['taxable_ordinary'].values
pref_tax = df['pref_tax'].values
niit = df['niit'].values
eitc = df['SPM_EITC'].values
actc = df['SPM_ACTC'].values
fedtax_ac = df['FEDTAX_AC'].values

# =============================================================================
# SURVEY WEIGHTS
# =============================================================================

if USE_WEIGHTS:
    marsupwt = df['MARSUPWT'].values
    weight_label = "weighted"
else:
    marsupwt = np.ones_like(df['MARSUPWT'].values)
    weight_label = "unweighted"

print(f"\nUsing {weight_label} specification (USE_WEIGHTS = {USE_WEIGHTS})")

# Observed revenue share
g_obs = np.sum(marsupwt * fedtax_ac) / np.sum(marsupwt * agi)
print(f"Observed g (FEDTAX_AC / AGI): {g_obs:.4f}")

# =============================================================================
# TAX FUNCTION
# =============================================================================

def compute_taxable_amounts(incomes, brackets):
    n = len(incomes)
    m = len(brackets) + 1
    taxable = np.zeros((n, m))
    remaining = incomes.astype(float).copy()
    taxable[:, 0] = np.minimum(remaining, brackets[0])
    remaining -= taxable[:, 0]
    for k in range(len(brackets) - 1):
        width = brackets[k + 1] - brackets[k]
        taxable[:, k + 1] = np.minimum(remaining, width)
        remaining -= taxable[:, k + 1]
    taxable[:, -1] = np.maximum(remaining, 0.0)
    return taxable

taxable = compute_taxable_amounts(taxable_ordinary, BRACKETS)

# =============================================================================
# OPTIMIZATION
# =============================================================================

effective_weight = (marsupwt * agi) / np.sum(marsupwt * agi)
n_rates = len(BRACKETS) + 1
x_stat = np.array([0.10, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37])

def make_revenue_constraint(g):
    def revenue_constraint(x):
        return np.sum(marsupwt * (taxable @ x - eitc - actc + pref_tax + niit)) \
               - g * np.sum(marsupwt * agi)
    return revenue_constraint

def optimize_tax_rates(sigma, g, x0=None):
    if x0 is None:
        x0 = x_stat.copy()

    constraints = [{'type': 'eq', 'fun': make_revenue_constraint(g)}]

    def objective_and_grad(x, sigma):
        y = agi - taxable @ x + eitc + actc - pref_tax - niit
        y = np.maximum(y, 1e-6)

        if sigma == 1.0:
            util = np.dot(effective_weight, np.log(y))
            grad = -np.dot(effective_weight / y, taxable)
        else:
            util = np.dot(effective_weight, y**(1.0 - sigma) / (1.0 - sigma))
            grad = -np.dot(effective_weight * (y ** (-sigma)), taxable)

        return -util, -grad

    def objective_hess(x):
        y = agi - taxable @ x + eitc + actc - pref_tax - niit
        y = np.maximum(y, 1e-6)

        if sigma == 1.0:
            second_deriv_weights = effective_weight / (y ** 2)
        else:
            second_deriv_weights = effective_weight * sigma * (y ** (-(sigma + 1)))

        return taxable.T @ (second_deriv_weights[:, None] * taxable)

    result = minimize(
        fun=lambda x: objective_and_grad(x, sigma),
        x0=x0,
        method='trust-constr',
        jac=True,
        hess=objective_hess,
        constraints=constraints,
        options=SOLVER_OPTIONS
    )
    return result

# =============================================================================
# BRACKET WEIGHTS
# =============================================================================

ind_tax_actual = taxable @ x_stat
bracket_bounds = [0] + list(BRACKETS) + [np.inf]
n_brackets = len(BRACKETS) + 1

total_pop_weight = np.sum(marsupwt)
total_tax_weighted = np.sum(marsupwt * ind_tax_actual)

pi_k = np.zeros(n_brackets)
pi_k_tax = np.zeros(n_brackets)

for k in range(n_brackets):
    lower = bracket_bounds[k]
    upper = bracket_bounds[k + 1]
    in_bracket = (taxable_ordinary >= lower) & (taxable_ordinary < upper)
    pi_k[k] = np.sum(marsupwt[in_bracket]) / total_pop_weight
    pi_k_tax[k] = np.sum(marsupwt[in_bracket] * ind_tax_actual[in_bracket]) / total_tax_weighted

assert np.isclose(np.sum(pi_k), 1.0)
assert np.isclose(np.sum(pi_k_tax), 1.0)

print(f"\nBracket population weights: {pi_k.round(4)}")

# =============================================================================
# WAAD FUNCTION
# =============================================================================

def compute_waad(optimal_rates):
    rate_errors = np.abs(optimal_rates - x_stat)
    waad_pop = np.sum(pi_k * rate_errors) * 100
    waad_tax = np.sum(pi_k_tax * rate_errors) * 100
    return waad_pop, waad_tax

# =============================================================================
# PART 1: DETAILED RESULTS FOR g = PRINT_DETAILED_G
# =============================================================================

print("\n" + "="*80)
print(f"DETAILED RESULTS FOR g = {PRINT_DETAILED_G:.3f} ({weight_label})")
print("="*80)

detailed_results = []
x0 = x_stat.copy()

for sigma in DETAILED_SIGMA_VALUES:
    result = optimize_tax_rates(sigma, PRINT_DETAILED_G, x0=x0)
    if result.success:
        waad_pop, waad_tax = compute_waad(result.x)
        detailed_results.append({
            'sigma': sigma,
            'waad_pop': waad_pop,
            'waad_tax': waad_tax,
            'rates': result.x
        })
        x0 = result.x
    else:
        print(f"  Failed at sigma = {sigma:.3f}")

print(f"{'Sigma':>8} {'WAAD_P':>10} {'WAAD_T':>10} {'B1':>8} {'B2':>8} {'B3':>8} {'B4':>8} {'B5':>8} {'B6':>8} {'B7':>8}")
print("-"*90)
for d in detailed_results:
    r = d['rates']
    print(f"{d['sigma']:8.3f} {d['waad_pop']:9.2f}% {d['waad_tax']:9.2f}% "
          f"{r[0]:8.3f} {r[1]:8.3f} {r[2]:8.3f} {r[3]:8.3f} "
          f"{r[4]:8.3f} {r[5]:8.3f} {r[6]:8.3f}")
print("-"*90)

# =============================================================================
# PART 2: JOINT SEARCH OVER g AND sigma
# =============================================================================

results = []

print("\n" + "="*80)
print(f"JOINT SEARCH OVER g AND sigma ({weight_label})")
print("="*80)

for g in G_GRID:
    print(f"\nProcessing g = {g:.3f}...")

    best_pop = {'waad': np.inf, 'g': g, 'sigma': None, 'waad_pop': None, 'waad_tax': None, 'rates': None}
    best_tax = {'waad': np.inf, 'g': g, 'sigma': None, 'waad_pop': None, 'waad_tax': None, 'rates': None}

    x0 = x_stat.copy()

    for sigma in SIGMA_GRID:
        result = optimize_tax_rates(sigma, g, x0=x0)

        if result.success:
            waad_pop, waad_tax = compute_waad(result.x)

            if waad_pop < best_pop['waad']:
                best_pop.update({
                    'waad': waad_pop,
                    'sigma': sigma,
                    'waad_pop': waad_pop,
                    'waad_tax': waad_tax,
                    'rates': result.x
                })

            if waad_tax < best_tax['waad']:
                best_tax.update({
                    'waad': waad_tax,
                    'sigma': sigma,
                    'waad_pop': waad_pop,
                    'waad_tax': waad_tax,
                    'rates': result.x
                })

            x0 = result.x

    if best_pop['sigma'] is not None:
        results.append(best_pop)
        results.append(best_tax)
        print(f"  Best pop: sigma = {best_pop['sigma']:.3f}, WAAD_P = {best_pop['waad_pop']:.2f}%, WAAD_T = {best_pop['waad_tax']:.2f}%")
        print(f"  Best tax: sigma = {best_tax['sigma']:.3f}, WAAD_P = {best_tax['waad_pop']:.2f}%, WAAD_T = {best_tax['waad_tax']:.2f}%")
    else:
        print(f"  No successful optimization for g = {g:.3f}")

# =============================================================================
# GLOBAL BEST
# =============================================================================

if results:
    global_pop = min(results, key=lambda x: x['waad_pop'])
    global_tax = min(results, key=lambda x: x['waad_tax'])

    print("\n" + "="*80)
    print(f"GLOBAL BEST RESULTS ({weight_label})")
    print("="*80)
    print(f"\nPopulation-weighted criterion:")
    print(f"  g = {global_pop['g']:.3f}")
    print(f"  sigma = {global_pop['sigma']:.3f}")
    print(f"  WAAD_P = {global_pop['waad_pop']:.2f}%")
    print(f"  WAAD_T = {global_pop['waad_tax']:.2f}%")
    print(f"  Rates = {np.round(global_pop['rates'], 3)}")

    print(f"\nTax-weighted criterion:")
    print(f"  g = {global_tax['g']:.3f}")
    print(f"  sigma = {global_tax['sigma']:.3f}")
    print(f"  WAAD_P = {global_tax['waad_pop']:.2f}%")
    print(f"  WAAD_T = {global_tax['waad_tax']:.2f}%")
    print(f"  Rates = {np.round(global_tax['rates'], 3)}")
