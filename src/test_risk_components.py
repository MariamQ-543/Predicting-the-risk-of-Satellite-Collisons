import pandas as pd
from sklearn.metrics import r2_score
from load_data import load_csv

test = load_csv("esa_test.csv")

# Test if risk = max_risk_estimate + max_risk_scaling
test['combined'] = test['max_risk_estimate'] + test['max_risk_scaling']

r2_combined = r2_score(test['risk'], test['combined'])
print(f"R² (estimate + scaling): {r2_combined:.4f}")

# Also test other combinations
r2_estimate_only = r2_score(test['risk'], test['max_risk_estimate'])
r2_scaling_only = r2_score(test['risk'], test['max_risk_scaling'])

print(f"R² (estimate only): {r2_estimate_only:.4f}")
print(f"R² (scaling only): {r2_scaling_only:.4f}")