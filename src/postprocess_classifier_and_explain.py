######################################################################################################
# Author: Ellen Sarauer                                                                              #
# Affiliation: German Aerospace Center (DLR)                                                         #
# Filename: postprocess_classifier_and_explain.py                                                    #
######################################################################################################
# In this script we postprocess the Microphysics Trigger Classifier Model.                           #
# We calculate Shapley values, generate a summary plot and plot the confusion matrix.                #
# For more information, please check Methodology section in our paper.                               #
######################################################################################################

# Import
import numpy as np
import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, Normalize
import shap
from build_classifier_model import TriggerClassifier

# Load data
data_path = "/work/bd1179/b309246/phd_thesis/sarauer23_microphysics_parametrization/pytorch_nextgems/data/"
set_test = np.load(data_path + "df_nextgems_mig_subset_classify_test.npy")
X_test = set_test[:, 1:9]
y_test = set_test[:, 9]

# Standard scaling
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

# Convert data to tensors
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Load the trained model
model = TriggerClassifier()
model.load_state_dict(torch.load('/work/bd1179/b309246/phd_thesis/sarauer23_microphysics_parametrization/pytorch_nextgems/models/classify_model.pth'))
model.eval()

# Predict model
def predict_pytorch(input_data):
    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    return output_tensor.numpy()

# Select a subset of 1000 samples for explainability
indices = np.random.choice(len(X_test_scaled), 1000, replace=False)
X_test_subset = X_test_scaled[indices]
y_test_subset = y_test[indices]

# Calculate Shapley values
explainer = shap.KernelExplainer(predict_pytorch, X_test_subset)
shap_values = explainer.shap_values(X_test_subset)
shap_values_array = np.array(shap_values).reshape((1000, 8))

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
    cbar.set_label('Standardized feature value', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    ticks = np.linspace(min_val, max_val, 5)
    cbar.set_ticks(ticks)
    tick_labels = [f'{tick:.2f}' for tick in ticks]
    cbar.set_ticklabels(tick_labels)
    plt.xlabel('Shapley value', fontsize=18)
    plt.ylabel('')
    plt.grid(True, linestyle='--', alpha=0.7)
    for label in ax.get_yticklabels():
        label.set_fontsize(18)
    plt.tight_layout(pad=1.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Feature names
input_features = ["air pressure", "temperature", "water vapor mmr", "cloud water mmr", "cloud ice mmr", "rain mmr", "snow mmr", "graupel mmr"]

# Generate and save Shapley summary plot
shap_summary_path = "/work/bd1179/b309246/phd_thesis/sarauer23_microphysics_parametrization/pytorch_nextgems/plots/shapley/explain_classification_subset.png"
save_shap_summary_plot(
    shap_values_array, 
    features=X_test_subset, 
    feature_names=input_features, 
    save_path=shap_summary_path
)

# Plot confusion matrix
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    y_test_pred = (test_outputs > 0.5).float()
cm = confusion_matrix(y_test_tensor.numpy(), y_test_pred.numpy())
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
confusion_matrix_path = "/work/bd1179/b309246/phd_thesis/sarauer23_microphysics_parametrization/pytorch_nextgems/plots/combine_scatter_explain/confusion_matrix.png"
plt.figure(figsize=(10, 8))
colors = ["#D0E5F2", "#A0C4E9", "#70A1D7", "#4081C6", "#003C7A"]
cmap = mcolors.LinearSegmentedColormap.from_list("custom_blues", colors)
ax = sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap=cmap,
                 xticklabels=['Not Triggered', 'Triggered'],
                 yticklabels=['Not Triggered', 'Triggered'],
                 cbar=True, annot_kws={"size": 30}, linewidths=.5, square=True,
                 cbar_kws={"shrink": 0.6, "aspect": 10, 'label': 'Proportion of Samples'})

# Formatting
ax.set_xlabel('Predicted Label', fontsize=30, labelpad=10)
ax.set_ylabel('True Label', fontsize=30, labelpad=10)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=24)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=24)
colorbar = ax.collections[0].colorbar
colorbar.set_label('Proportion of Samples', fontsize=30)
colorbar.ax.tick_params(labelsize=24)
plt.tight_layout()
plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
plt.close()

# Combine the plots into one figure
def combine_plots(confusion_matrix_path, shap_summary_path, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8), gridspec_kw={'width_ratios': [1.2, 2]})
    img_cm = plt.imread(confusion_matrix_path)
    ax1.imshow(img_cm)
    ax1.axis('off')
    img_shap = plt.imread(shap_summary_path)
    ax2.imshow(img_shap)
    ax2.axis('off')
    plt.suptitle('Microphysics trigger classifier', fontsize=30, fontweight='bold')
    plt.subplots_adjust(wspace=0.05)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# Save the final figure
combined_plot_path = "/work/bd1179/b309246/phd_thesis/sarauer23_microphysics_parametrization/pytorch_nextgems/plots/combine_scatter_explain/combined_plot.png"
combine_plots(confusion_matrix_path, shap_summary_path, combined_plot_path)
print(f"Combined plot saved to {combined_plot_path}")
