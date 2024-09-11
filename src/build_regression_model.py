######################################################################################################
# Author: Ellen Sarauer                                                                              #
# Affiliation: German Aerospace Center (DLR)                                                         #
# Filename: build_regression_model.py                                                                #
######################################################################################################
# In this script we define the architecture of the Microphysics Regression Model.                    #
# The class RegressionMLP defines the number of layers, dropout and implements residual connection.  #
# The class  MassPositivityConservationLoss implements the physics-constraints.                      #
# For more information, please check Methodology section in our paper.                               #
######################################################################################################

# Import
import torch
import torch.nn as nn
import torch.nn.functional as F

# Regression MLP
class RegressionMLP(nn.Module):
    
    def __init__(self, max_nodes = 512, n_inputs = 8, n_outputs = 7):
        super(RegressionMLP, self).__init__()
        self.shared_network = nn.Sequential(
            nn.Linear(n_inputs, max_nodes),
            nn.ReLU(),
            nn.Linear(max_nodes, max_nodes),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(max_nodes, max_nodes),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(max_nodes, n_outputs)
        self.input_adjuster = nn.Linear(n_inputs, n_outputs)
    
    def forward(self, x):
        x_shared = self.shared_network(x)
        outputs = self.output_layer(x_shared)
        input_adjusted = self.input_adjuster(x)
        if outputs.size(1) != input_adjusted.size(1):
            raise ValueError("Output dimensions and adjusted input dimensions do not match.")
        final_outputs = outputs + input_adjusted
        return final_outputs


# Physics-constraining
class MassPositivityConservationLoss(nn.Module):
    
    def __init__(self, mean_y, std_y, mean_x, std_x):
        super(MassPositivityConservationLoss, self).__init__()
        self.mean_y = mean_y
        self.std_y = std_y
        self.mean_x = mean_x
        self.std_x = std_x

    def forward(self, y_true, y_pred, x_inputs):
        
        # model time step
        time_step = 40

        # Rescale to physical values
        y_pred_rescaled = y_pred * self.std_y + self.mean_y
        x_inputs_rescaled = x_inputs[:, :] * self.std_x + self.mean_x
        mse_loss = nn.MSELoss()(y_true * self.std_y + self.mean_y, y_pred_rescaled)
        
        # Constraint 1: Enforce X + tend_X * time_step >= 0
        constraints = [
            x_inputs_rescaled[:, i] + y_pred_rescaled[:, i] * time_step
            for i in range(7)
        ]
        constraint_penalty = sum([torch.mean(F.relu(-constraint)) for constraint in constraints])

        # Constraint 2: Enforce conservation of total q*
        sum_q_initial = torch.sum(x_inputs_rescaled[:, :], dim=1)
        sum_tend_q = torch.sum(y_pred_rescaled[:, :] * time_step, dim=1)
        conservation_penalty = torch.mean((sum_q_initial - (sum_q_initial + sum_tend_q)) ** 2)
        total_loss = mse_loss + constraint_penalty + conservation_penalty
        
        return total_loss

if __name__ == "__main__":
    model = RegressionMLP()
    model.eval()
