######################################################################################################
# Author: Ellen Sarauer                                                                              #
# Affiliation: German Aerospace Center (DLR)                                                         #
# Filename: train_classifier_model.py                                                                #
######################################################################################################
# In this script we train Microphysics Trigger Classifier Model.                                     #
# We scale our preprocessed inputs and call our model from build_classifier_model.py.                #
# We train our model and store it in the models folder.                                              #
# For more information, please check Methodology section in our paper.                               #
######################################################################################################

# Import
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_curve, auc
from torch.optim.lr_scheduler import StepLR  
from build_classifier_model import TriggerClassifier

# Load data
data_path = "path/to/data/"
out_path = "/path/to/out/"
set_train = np.load(data_path + "df_nextgems_mig_subset_classify_train.npy")
set_val = np.load(data_path + "df_nextgems_mig_subset_classify_val.npy")
set_test = np.load(data_path + "df_nextgems_mig_subset_classify_test.npy")

# Extract features and labels
X_train = set_train[:, 1:9]
y_train = set_train[:, 9]
X_val = set_val[:, 1:9]
y_val = set_val[:, 9]
X_test = set_test[:, 1:9]
y_test = set_test[:, 9]

# Standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# Load model
model = TriggerClassifier()

# Hyperparameters
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
patience = 5  # Number of epochs to wait for improvement before stopping
best_val_loss = float('inf')
patience_counter = 0

# Training the model
num_epochs = 20
train_losses = []
val_losses = []
print("start training")

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * inputs.size(0)
    epoch_loss /= len(train_loader.dataset)
    train_losses.append(epoch_loss)
    model.eval()
    
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor).item()
        val_losses.append(val_loss)
    
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0  # Reset counter
        torch.save(model.state_dict(), out_path + 'classify_model_20.pth')  # Save the best model
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered!")
        break

    scheduler.step()

# Load the best model
model.load_state_dict(torch.load( out_path + 'classify_model_20.pth'))

# Plot training & validation loss values
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss History')
plt.legend()
plt.savefig( out_path + "classify_loss.png", dpi=300, bbox_inches='tight')
plt.close()

# Evaluate on test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    y_test_pred = (test_outputs > 0.5).float()

# Calculate metrics
f1 = f1_score(y_test_tensor.numpy(), y_test_pred.numpy())
roc_auc = roc_auc_score(y_test_tensor.numpy(), test_outputs.numpy())

print(f"F1 Score: {f1:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Plot confusion matrix with ratios
cm = confusion_matrix(y_test_tensor.numpy(), y_test_pred.numpy())
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
            xticklabels=['Not Triggered', 'Triggered'], 
            yticklabels=['Not Triggered', 'Triggered'],
            cbar=True, annot_kws={"size": 14}, linewidths=.5, square=True, 
            cbar_kws={"shrink": 0.75, "aspect": 15})
plt.xlabel('Predicted Label', fontsize=18, labelpad=10)
plt.ylabel('True Label', fontsize=18, labelpad=10)
plt.title('Normalized Confusion Matrix of Microphysics Trigger Classifier', fontsize=20,  fontweight='bold', pad=20)
plt.xticks(fontsize=14, rotation=45, ha="right", rotation_mode="anchor")
plt.yticks(fontsize=14, rotation=0)
plt.tight_layout()
plt.savefig(out_path + "confusion_matrix_normalized.png", dpi=300, bbox_inches='tight')

# Plot the ROC curve
fpr, tpr, _ = roc_curve(y_test_tensor.numpy(), y_test_pred.numpy())
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate', fontsize=16, labelpad=10)
plt.ylabel('True Positive Rate', fontsize=16, labelpad=10)
plt.title('ROC Curve of Microphysics Trigger Classifier', fontsize=18, pad=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower right', fontsize=14)
plt.tight_layout()
plt.savefig(out_path + "roc_auc.png", dpi=300, bbox_inches='tight')