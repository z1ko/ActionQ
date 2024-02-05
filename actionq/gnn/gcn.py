import torch
import torch.nn as nn
import torch_geometric.nn as gnn

class GCNLayer(nn.Module):
    def __init__(
        self,
        skeleton,
        input_dim,
        output_dim,
        droput
    ):
        super().__init__()

        self.skeleton = skeleton

        #self.norm = nn.LayerNorm(input_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=droput)
        self.conv = gnn.dense.DenseGCNConv(
            in_channels=input_dim,
            out_channels=output_dim
        )

    def forward(self, x):
        #x = self.norm(x)
        x = self.conv(x, self.skeleton)
        x = self.activation(x)
        y = self.dropout(x)
        return y


class GCNSkeleton(nn.Module):
    def __init__(
        self, 
        skeleton,   # Adj. matrix of the skeleton
        input_dim,
        output_dim,
        layers_count,
        dropout,
        hops=1
    ):
        super().__init__()

        # Create multi-hop adj. matrices (hops, N, N)
        self.A = skeleton.unsqueeze(0)[0], #self.create_adj_matrix(skeleton, hops)

        # Basic GCN model
        self.gnn_layers = nn.ModuleList()
        for _ in range(layers_count):
            self.gnn_layers.append(
                GCNLayer(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    skeleton=self.A,
                    droput=dropout
                )
            )

    def create_adj_matrix(self, skeleton, hops):
        
        A = skeleton
        A2 = A @ skeleton
        A3 = A2 @ skeleton

        results = [skeleton]
        for i in range(1, hops):
            adj = results[i] @ skeleton
            adj = torch.where(adj == i + 1, 1, 0)
            results.append(adj)

        return torch.stack(results, dim=0)

    def forward(self, x): # (B, J, F)
        for layer in self.gnn_layers:
            x = layer(x)

        return x # (B, J, F')
