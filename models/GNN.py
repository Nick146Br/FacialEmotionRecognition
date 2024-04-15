import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SSGConv

class SSG(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_hidden_layers, out_channels, device):
        super(SSG, self).__init__()
        
        # self.conv1 = GATConv(in_channels, hidden_channels, add_self_loops=False, heads=heads, concat=True, dropout=dropout)
        # self.device = device
        # self.convM = [GATConv(hidden_channels*heads, hidden_channels*heads,concat=False, add_self_loops=False ,heads=heads, dropout=dropout) for _ in range(num_hidden_layers)]
        self.conv1 = SSGConv(in_channels, hidden_channels, alpha=0.3)
        self.device = device
        self.convM = [SSGConv(hidden_channels, hidden_channels, alpha=0.3) for _ in range(num_hidden_layers)]
        self.fc1 = nn.Linear(2752, 2752//2)
        self.fc2 = nn.Linear(2752//2, out_channels)
        self.relu = torch.nn.functional.relu
        self.dropout = nn.Dropout(p=0.5)
        self.leaky_relu = torch.nn.functional.leaky_relu
        
    def forward(self, x, edge_index, edge_weight, batch):
        # x = self.normalization(x)
        x = self.conv1(x, edge_index, edge_weight)
        x = torch.tanh(x)
        
        for conv in self.convM:
            conv.to(self.device)
            x = conv(x, edge_index, edge_weight)
            x = torch.tanh(x)
            
        # x, edge_index, edge_weight, batch, _, _ = self.top_k(x, edge_index, edge_attr=edge_weight, batch=batch)
        # x = global_mean_pool(x, batch)
        
        x = self.dropout(x)
        #num elementos diferentes no batch
        num_elements = len(set(batch.tolist()))
        x = x.reshape(num_elements, -1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
