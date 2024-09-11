######################################################################################################
# Author: Ellen Sarauer                                                                              #
# Affiliation: German Aerospace Center (DLR)                                                         #
# Filename: postprocess_regression_and_explain.py                                                    #
######################################################################################################
# In this script we postprocess the Microphysics Regression Model.                                   #
# We calculate Shapley values, generate a summary plot and ML vs ground truth in a scatter plot.     #                #
# For more information, please check Methodology section in our paper.                               #
######################################################################################################

# Import
import shap
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm
from matplotlib.ticker import FuncFormatter
from PIL import Image

# Functions for handling expoential data in axis
def factor_formatter(x, _):
    if x == 0:
        return '0.00'
    else:
        factor = x / 10**int(np.log10(abs(x)))
        return f'{factor:.2f}'
def get_common_exponent(data):
    if np.all(data == 0):
        return 0
    return int(np.floor(np.log10(np.max(np.abs(data)))))

# Load model and data
model_path = "path/to/model/"
ann_model = torch.load(model_path + "regression_model.pt")
ann_model.eval()
data_path = "path/to/data/"
set_test = np.load(data_path + "df_nextgems_mig_subset_regression_test.npy")

# Standard scaling
scaler = StandardScaler()
set_test_scaled = scaler.fit_transform(set_test)
inputset_test = torch.tensor(set_test_scaled[:, 1:9], dtype=torch.float32)
outputset_test = torch.tensor(set_test_scaled[:, 9:], dtype=torch.float32)
input_features = ["air pressure", "temperature", "water vapor mmr", "cloud water mmr", "cloud ice mmr", "rain mmr", "snow mmr", "graupel mmr"]

# Predict model
def predict_pytorch(input_data):
    with torch.no_grad():
        return ann_model(torch.tensor(input_data, dtype=torch.float32)).numpy()

# Select a subset of 1000 samples for explainability
inputset_test_sampled = shap.sample(inputset_test.numpy(), 1000)

# Calculate Shapley values
explainer = shap.KernelExplainer(predict_pytorch, inputset_test_sampled)
shap_values = explainer.shap_values(inputset_test_sampled)

# Shapley summary plotting
def save_shap_summary_plot(shap_values, features, feature_names, save_path):
    plt.figure(figsize=(12, 9))
    colors = ['#0033FF', '#66CCFF', '#FF9999', '#FF3333']
    custom_rdbu = LinearSegmentedColormap.from_list('custom_rdbu', [(0, colors[0]), (0.4, colors[1]), (0.6, colors[2]), (1, colors[3])])
    min_val = np.min(shap_values)
    max_val = np.max(shap_values)
    shap.summary_plot(shap_values, features=features, feature_names=feature_names, show=False, cmap=custom_rdbu, color_bar=False)
    ax = plt.gca()
    ax.tick_params(axis='x', labelsize=14)
    norm = Normalize(vmin=min_val, vmax=max_val)
    sm = plt.cm.ScalarMappable(cmap=custom_rdbu, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
    cbar.set_label('standardized feature value', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    ticks = np.linspace(min_val, max_val, 5)
    cbar.set_ticks(ticks)
    tick_labels = [f'{tick:.2f}' for tick in ticks]
    cbar.set_ticklabels(tick_labels) 
    plt.xlabel('Shapley value', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    for label in ax.get_yticklabels():
        label.set_fontsize(18)
    plt.tight_layout(pad=1.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Output names
output_nodes = ["tend_ta_mig", "tend_qv_mig", "tend_qc_mig", "tend_qi_mig", "tend_qr_mig", "tend_qs_mig", "tend_qg_mig"]

# Generate and save Shapley summary plot for each output
out_path = "path/to/out/"
for i, node in enumerate(output_nodes):
    save_shap_summary_plot(shap_values[:, :, i], inputset_test_sampled, input_features, out_path + f"explain_{node}.pdf")

# Regression scatter plots
model = torch.load(model_path + "regression_model_long.pt")
model.eval()
X_val = set_test_scaled[:, 1:9]
y_val = set_test_scaled[:, 9:16]
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
y_pred_tensor = model(X_val_tensor)
y_pred = y_pred_tensor.detach().numpy()
y_val_unscaled = scaler.inverse_transform(np.hstack((np.zeros((y_val.shape[0], 9)), y_val)))[:, 9:16]
y_pred_unscaled = scaler.inverse_transform(np.hstack((np.zeros((y_pred.shape[0], 9)), y_pred)))[:, 9:16]

# Variables information
variables_info = [
    ("tend_ta_mig", "tendency of temperature", "T$\cdot$ s$^{-1}$"),
    ("tend_qv_mig", "tendency of water vapor mmr", "kg $\cdot$ kg$^{-1}$ $\cdot$ s$^{-1}$"),
    ("tend_qc_mig", "tendency of cloud water mmr", "kg $\cdot$ kg$^{-1}$ $\cdot$ s$^{-1}$"),
    ("tend_qi_mig", "tendency of cloud ice mmr", "kg $\cdot$ kg$^{-1}$ $\cdot$ s$^{-1}$"),
    ("tend_qr_mig", "tendency of rain mmr", "kg $\cdot$ kg$^{-1}$ $\cdot$ s$^{-1}$"),
    ("tend_qs_mig", "tendency of snow mmr", "kg $\cdot$ kg$^{-1}$ $\cdot$ s$^{-1}$"),
    ("tend_qg_mig", "tendency of graupel mmr", "kg $\cdot$ kg$^{-1}$ $\cdot$ s$^{-1}$"),
]

# Combine the plots into one figure
for i, (var_name, var_title, var_unit) in enumerate(variables_info):
    y_test_var = y_val_unscaled[:, i]
    y_prediction_var = y_pred_unscaled[:, i]
    r2 = r2_score(y_test_var, y_prediction_var)
    p25_test, p975_test = np.percentile(y_test_var, [0.001, 99.999])
    p25_pred, p975_pred = np.percentile(y_prediction_var, [0.001, 99.999])
    mask = (y_test_var >= p25_test) & (y_test_var <= p975_test) & (y_prediction_var >= p25_pred) & (y_prediction_var <= p975_pred)
    y_test_var_filtered = y_test_var[mask]
    y_prediction_var_filtered = y_prediction_var[mask]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 7), gridspec_kw={'width_ratios': [1, 2.4]})  # Make the right subplot larger
    hist = ax1.hist2d(y_test_var, y_prediction_var, bins=70, cmap='viridis', norm=LogNorm())
    cbar = plt.colorbar(hist[3], ax=ax1)
    cbar.set_label('data density', fontsize=24)
    cbar.ax.tick_params(labelsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    min_val = min(np.min(y_test_var), np.min(y_prediction_var))
    max_val = max(np.max(y_test_var), np.max(y_prediction_var))
    ax1.set_xlim([min_val, max_val])
    ax1.set_ylim([min_val, max_val])
    ticks = np.linspace(min_val, max_val, 5)
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)
    x_exp = get_common_exponent(y_test_var_filtered)
    ax1.xaxis.set_major_formatter(FuncFormatter(factor_formatter))
    ax1.yaxis.set_major_formatter(FuncFormatter(factor_formatter))  
    ax1.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, label='optimal fit')
    ax1.set_xlabel(f'ground truth $\\times 10^{{{x_exp}}}$', fontsize=24)
    ax1.set_ylabel(f'ML prediction $\\times 10^{{{x_exp}}}$', fontsize=24)
    ax1.text(0.05, 0.95, f'$R^2 = {r2:.2f}$', transform=ax1.transAxes, fontsize=24, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax1.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
    ax1.legend(loc='lower right', fontsize=24)
    save_shap_summary_plot(shap_values[:, :, i], inputset_test_sampled, input_features, save_path=out_path + "temp_plot.png")
    with Image.open(out_path +"temp_plot.png") as img:
        img = img.convert("RGB")
        img.save(out_path +"resized_temp_plot.png")
    img = plt.imread(out_path +"resized_temp_plot.png")
    ax2.imshow(img)
    ax2.axis('off')
    fig.suptitle(f'Microphysics regression model for {var_title} [{var_unit}]', 
             fontsize=30, y=1.05, fontweight='bold') 
    plt.subplots_adjust(top=0.93, wspace=0.005)
    # Save the final figure
    plt.savefig(out_path + f"{var_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

