import numpy as np
from sklearn.metrics import r2_score
import xarray as xr

# Define varname
varnames = [
    "tend_qcli_mig", "tend_qclw_mig", "tend_qg_mig",
    "tend_qhus_mig", "tend_qr_mig", "tend_qs_mig", "tend_ta_mig"
]
r2_scores = []

# Load Data
for varname in varnames:
    true_path = f"/work/bd1179/b309246/experiments/r2b9_amip/coarse-grained-data/ml_tendencies/classical_mig/original_{varname}.nc"
    y_true = xr.open_dataset(true_path)
    ml_path = f"/work/bd1179/b309246/experiments/r2b9_amip/coarse-grained-data/ml_tendencies/ml_mig/ml_regression_{varname}.nc"
    y_pred = xr.open_dataset(ml_path)
    true_values = y_true[varname].values.flatten()
    ml_values = y_pred[varname].values.flatten()
    r2 = r2_score(true_values, ml_values)
    r2_scores.append(r2)
    print(f"R^2 score for {varname}: {r2}")

# Calculate unified R^2 score
r2 = r2_score(true_values, ml_values)
print("Unified R^2 score:", r2)
