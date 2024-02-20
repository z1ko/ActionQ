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
        dropout,
        mode='learning',
        skeleton=None,
        **kwargs
    ):
        super().__init__()
        self.mode = mode

        self.initial = nn.Sequential(
            nn.Linear(joint_features, joint_expansion),
            nn.LeakyReLU(),
        )

        # NOTE: Use one aggregator for all joint time series
        self.temporal_layers = nn.ModuleList()
        for _ in range(temporal_layers_count):
            self.temporal_layers.append(
                LRULayer(
                    state_dim=joint_expansion,
                    activation='gelu',
                    dropout=dropout,
                    **kwargs
                )
            )

        # How to aggregate temporal output (BJ, L, F) -> (BJ, F')
        # TODO: Try different aggregators
        self.temporal_aggregator = lambda x: x[:, -1, :]  # torch.mean(x, 1)

        # self.skeleton = skeleton.unsqueeze(0)[0].to(torch.device('cuda'))
        # self.spatial_layers = nn.ModuleList()
        # for i in range(spatial_layers_count):
        #    self.spatial_layers.append(
        #        GCNLayer(
        #            input_dim=joint_expansion if i == 0 else 128,
        #            output_dim=128,
        #            skeleton=self.skeleton,
        #            droput=dropout
        #        )
        #    )

        # From joint data obtains exercise score
        # self.final = nn.Sequential(
        #    nn.Linear(joint_expansion, joint_expansion),
        #    nn.LeakyReLU(),
        #    nn.Linear(joint_expansion, output_dim),
        #    nn.Sigmoid()
        # )

        # NOTE: This should be replaced with a GNN or attention mechanism
        self.condenser = nn.Linear(joint_expansion, 128)
        self.final = nn.Sequential(
            nn.Linear(128 * joint_count, 128),
            nn.LeakyReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """ Input x shape: (B, L, J, F)
        """

        B, L, J, F = x.shape

        x = self.initial(x)

        # Process temporal sequence to (BJ, F)
        x = ein.rearrange(x, 'B L J F -> (B J) L F')
        for temporal_layer in self.temporal_layers:
            x = temporal_layer(x)

        x = self.temporal_aggregator(x)

        # Process spatial nodes (B, J, F) -> (B, J, F')
        # x = ein.rearrange(x, '(B J) F -> B J F', B=B, J=J)
        # for spatial_layer in self.spatial_layers:
        #    x = spatial_layer(x)

        # Concatenate all nodes representation
        x = self.condenser(x)  # (BJ, F) -> (BJ, F')
        x = ein.rearrange(x, '(B J) F -> B (J F)', B=B, J=J)
        # x = torch.sum(x, dim=1)  # (B F)
        y = self.final(x)

        return y

    def forward_with_state(self, x, state):
        """ Input x shape (J, F), state shape (T, J, S)
        """

        assert (self.mode == 'predict')

        x = self.initial(x)

        for i, temporal_layer in enumerate(self.temporal_layers):
            x, state[i] = temporal_layer.forward_with_state(x, state[i])

        # no temporal aggregator, just use last output
        x = self.condenser(x)
        x = ein.rearrange(x, 'J D -> (J D)')
        y = self.final(x)

        return y
