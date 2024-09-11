######################################################################################################
# Author: Ellen Sarauer                                                                              #
# Affiliation: German Aerospace Center (DLR)                                                         #
# Filename: build_classifier_model.py                                                                #
######################################################################################################
# In this script we define the architecture of the Microphysics Trigger Classifier Model.            #
# The class TriggerClassifier defines the number of layers and dropout.                              #
# For more information, please check Methodology section in our paper.                               #
######################################################################################################

# Import
import torch
import torch.nn as nn
import torch.optim as optim

# Define the TriggerClassifier model
class TriggerClassifier(nn.Module):

    def __init__(self, dropout_prob=0.1, max_nodes = 512, n_inputs = 8, n_outputs = 1):
        super(TriggerClassifier, self).__init__()
        self.fc1 = nn.Linear(n_inputs, max_nodes)
        self.fc2 = nn.Linear(max_nodes, max_nodes/2)
        self.fc3 = nn.Linear(max_nodes/2, n_outputs)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x) 
        x = self.sigmoid(self.fc3(x))  
        return x

if __name__ == "__main__":
    model = TriggerClassifier()
    model.eval()