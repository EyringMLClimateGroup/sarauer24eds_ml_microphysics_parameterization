######################################################################################################
# Author: Ellen Sarauer                                                                              #
# Affiliation: German Aerospace Center (DLR)                                                         #
# Filename: train_regression_model.py                                                                #
######################################################################################################
# In this script we train Microphysics Regression Model.                                             #
# We scale our preprocessed inputs and call our model from build_regression_model.py.                #
# We train our model and store it in the models folder.                                              #
# For more information, please check Methodology section in our paper.                               #
######################################################################################################

# Import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import tqdm
import copy
import matplotlib.pyplot as plt
from build_regression_model import RegressionMLP, MassPositivityConservationLoss

# Load data
data_path = "path/to/data/"
set_train = np.load(data_path+"df_nextgems_mig_subset_regression_train.npy")
set_val = np.load(data_path+"df_nextgems_mig_subset_regression_val.npy")
set_test = np.load(data_path+"df_nextgems_mig_subset_regression_test.npy")

# Apply standard scaling to training, validation, and test data
scaler = StandardScaler()
set_train_scaled = scaler.fit_transform(set_train)
set_val_scaled = scaler.transform(set_val)
set_test_scaled = scaler.transform(set_test)

# Extract means and standard deviations
mean = scaler.mean_
std = scaler.scale_

# Convert to torch tensors
mean_y = torch.tensor(mean[9:])
std_y = torch.tensor(std[9:])
mean_x = torch.tensor(mean[1:9])
std_x = torch.tensor(std[1:9])

# Choose whether to use custom loss or standard MSE
use_custom_loss = False  # Set to False to use MSELoss
if use_custom_loss:
    loss_fn = MassPositivityConservationLoss(mean_y, std_y, mean_x, std_x)
else:
    loss_fn = nn.MSELoss()

# Split into input and output sets
inputset_train = torch.tensor(set_train_scaled[:,1:9], dtype=torch.float32)
outputset_train = torch.tensor(set_train_scaled[:,9:], dtype=torch.float32)
inputset_val = torch.tensor(set_val_scaled[:,1:9], dtype=torch.float32)
outputset_val = torch.tensor(set_val_scaled[:,9:], dtype=torch.float32)
inputset_test = torch.tensor(set_test_scaled[:,1:9], dtype=torch.float32)
outputset_test = torch.tensor(set_test_scaled[:,9:], dtype=torch.float32)

# Load model
model = RegressionMLP()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training parameters
n_epochs = 30
batch_size = 256
batch_start = torch.arange(0, len(inputset_train), batch_size)

# Early stopping parameters
patience = 5  
best_mse = np.inf
best_weights = None
history = []
no_improvement_count = 0

# Training loop
for epoch in range(n_epochs):
    model.train()
    with tqdm.tqdm(range(0, len(inputset_train), batch_size), unit="batch", mininterval=0, disable=False) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            X_batch = inputset_train[start:start+batch_size]
            y_batch = outputset_train[start:start+batch_size]
            y_pred = model(X_batch)
            if use_custom_loss:
                loss = loss_fn(y_batch, y_pred, X_batch)
            else:
                loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
            bar.set_postfix(loss=loss.item())
    
    scheduler.step()
    model.eval()
    y_pred = model(inputset_val)

    if use_custom_loss:
        mse = loss_fn(outputset_val, y_pred, inputset_val)
    else:
        mse = loss_fn(y_pred, outputset_val)
    
    mse = float(mse)
    history.append(mse)
    
    if mse < best_mse:
        best_mse = mse
        best_weights = copy.deepcopy(model.state_dict())
        no_improvement_count = 0  
    else:
        no_improvement_count += 1
    if no_improvement_count >= patience:
        print(f"Early stopping at epoch {epoch}. Best MSE: {best_mse:.2f}")
        break

# Restore model to best weights
model.load_state_dict(best_weights)
print(f"Best MSE: {best_mse:.2f}")
print(f"Best RMSE: {np.sqrt(best_mse):.2f}")

# Calculate R2 score
y_pred_np = y_pred.detach().numpy()
outputset_val_np = outputset_val.numpy()
r2_scores = [r2_score(outputset_val_np[:, i], y_pred_np[:, i]) for i in range(outputset_val_np.shape[1])]
for i, score in enumerate(r2_scores):
    print(f"R2 score for output node {i}: {score:.2f}")

# Plot training history
plt.plot(history)
plt.xlabel("Epoch")
plt.ylabel("Validation MSE")
out_path = "path/to/out"
plt.savefig(out_path + 'regression_history.png')

# Save the model
torch.save(model, out_path + "regression_model_long.pt")