'''MISATO, a database for protein-ligand interactions
    Copyright (C) 2023  
                        Till Siebenmorgen  (till.siebenmorgen@helmholtz-munich.de)
                        Sabrina Benassou   (s.benassou@fz-juelich.de)
                        Filipe Menezes     (filipe.menezes@helmholtz-munich.de)
                        Erinç Merdivan     (erinc.merdivan@helmholtz-munich.de)

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software 
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, NNConv
from torch.nn import Sequential as Seq, Linear, ReLU

class GNN_MD(torch.nn.Module):
    """
    Advanced GNN model for MD data with Message Passing and Attention.
    Args:
    num_features (int): Number of features in the input data
    hidden_dim (int): Hidden dimension of the GNN model
    """
    def __init__(self, num_features, hidden_dim, heads=4):
        super(GNN_MD, self).__init__()
        
        # 1. Message Passing Layer (NNConv)
        # Edge attribute is 1D (distance). Map 1 -> num_features * hidden_dim
        nn_edge1 = Seq(Linear(1, 16), ReLU(), Linear(16, num_features * hidden_dim))
        self.conv1 = NNConv(num_features, hidden_dim, nn=nn_edge1, aggr='mean')
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        # 2. Attention Layer 1 (GATv2Conv)
        # Output dim is hidden_dim * 2. We use multiple heads.
        self_hidden_2 = hidden_dim * 2
        self.conv2 = GATv2Conv(hidden_dim, self_hidden_2 // heads, heads=heads, edge_dim=1)
        self.bn2 = nn.BatchNorm1d(self_hidden_2)
        
        # 3. Attention Layer 2 (GATv2Conv)
        self_hidden_3 = hidden_dim * 4
        self.conv3 = GATv2Conv(self_hidden_2, self_hidden_3 // heads, heads=heads, edge_dim=1)
        self.bn3 = nn.BatchNorm1d(self_hidden_3)
        
        # 4. Standard GCN Layer
        self.conv4 = GCNConv(hidden_dim * 4, hidden_dim * 4)
        self.bn4 = nn.BatchNorm1d(hidden_dim * 4)
        
        # 5. Standard GCN Layer
        self.conv5 = GCNConv(hidden_dim * 4, hidden_dim * 8)
        self.bn5 = nn.BatchNorm1d(hidden_dim * 8)
        
        # Fully Connected Readout
        self.fc1 = nn.Linear(hidden_dim * 8, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, 1)


    def forward(self, data):
        # Edge attributes are distance. Shape must be (num_edges, 1) for NNConv and edge_dim in GATv2Conv
        edge_attr = data.edge_attr.view(-1, 1)
        
        x = self.conv1(data.x, data.edge_index, edge_attr)
        x = F.relu(x)
        x = self.bn1(x)
        
        x = self.conv2(x, data.edge_index, edge_attr)
        x = F.relu(x)
        x = self.bn2(x)
        
        x = self.conv3(x, data.edge_index, edge_attr)
        x = F.relu(x)
        x = self.bn3(x)
        
        # For GCNConv, it accepts 1D edge_weight if passed, or we can just pass edge_attr.view(-1)
        edge_weight = data.edge_attr.view(-1)
        x = self.conv4(x, data.edge_index, edge_weight)
        x = self.bn4(x)
        x = F.relu(x)
        
        x = self.conv5(x, data.edge_index, edge_weight)
        x = self.bn5(x)
        x = F.relu(x)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.25, training=self.training)
        return self.fc2(x).view(-1)

