import torch
import torch.nn as nn

class GCNSkeletonConv(nn.Module):
    def __init__(
        self, 
        skeleton,   # Adj. matrix of the skeleton
        input_dim, 
        output_dim,
        hops,
    ):
        super().__init__()

        # Create multi-hop adj. matrices (hops, N, N)
        self.A = self.create_adj_matrix(skeleton, hops)

        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.bias = nn.Parameter(torch.empty(output_dim))

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
        pass
