import torch
import torch.nn as nn
import einops as ein

from actionq.rnn.lru import LRULayer
from actionq.gnn.gcn import GCNLayer

class LRUModel(nn.Module):
    """Simple LRU module for temporal skeleton data
    """

    def __init__(
        self,
        joint_count,
        joint_features,
        joint_expansion,
        output_dim,
        temporal_layers_count,
        spatial_layers_count,
        skeleton,
        dropout
    ):
        super().__init__()


        self.initial_encoder = nn.Sequential(
            nn.Linear(joint_features, joint_expansion)
        )

        # NOTE: Use one aggregator for all joint time series
        self.temporal_layers = nn.ModuleList()
        for _ in range(temporal_layers_count):
            self.temporal_layers.append(
                LRULayer(
                    state_dim=joint_expansion,
                    dropout=dropout
                )
            )

        # How to aggregate temporal output (BJ, L, F) -> (BJ, F)
        # TODO: Try different aggregators
        self.temporal_aggregator = lambda x: torch.mean(x, 1)

        self.skeleton = skeleton.unsqueeze(0)[0].to(torch.device('cuda'))

        self.spatial_layers = nn.ModuleList()
        for i in range(spatial_layers_count):
            self.spatial_layers.append(
                GCNLayer(
                    input_dim=joint_expansion if i == 0 else 128,
                    output_dim=128,
                    skeleton=self.skeleton,
                    droput=dropout
                )
            )

        # From joint data obtains exercise score
        self.regressor = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

        #self.regressor = nn.Sequential(
        #    nn.Linear(joint_expansion*joint_count, joint_expansion*joint_count),
        #    nn.GELU(),
        #    nn.Linear(joint_expansion*joint_count, output_dim),
        #    nn.Sigmoid()
        #)

    def forward(self, x):
        """ Input x shape: (B, L, J, F)
        """

        B, L, J, F = x.shape

        x = self.initial_encoder(x)

        # Process temporal sequence to (BJ, F)
        x = ein.rearrange(x, 'B L J F -> (B J) L F')
        for temporal_layer in self.temporal_layers:
            x = temporal_layer(x)
        x = self.temporal_aggregator(x)

        # Process spatial nodes (B, J, F) -> (B, J, F')
        x = ein.rearrange(x, '(B J) F -> B J F', B=B, J=J)
        for spatial_layer in self.spatial_layers:
            x = spatial_layer(x)

        # Concatenate all nodes representation
        #x = ein.rearrange(x, 'B J F -> B (J F)')
        x = torch.sum(x, dim = 1)
        y = self.regressor(x)

        return y
