import torch
import torch.nn as nn

# TODO: Use simpler S4 block
from actionq.model.s4 import AQS4Block

class SingleJointTemporalBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        layers_count,
        dropout,
        aggregator='mean',
    ):
        super().__init__()

        # Initial node expansion
        # NOTE: Use a MLP (?)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )

        temporal_layers = []
        for _ in range(layers_count):
            temporal_layers.append(
                # NOTE: Maybe other SSMs can obtain better results: MAMBA, S5, H3
                # The final activation should be capped, otherwise we cant join representations
                AQS4Block( 
                    # TODO: How to initialize?
                    output_dim,
                    mode='s4d',         # Test different initialization modes.
                    dropout=dropout,
                    transposed=False,   # Accepted shape: (B, L, F)
                    lr=0.001,
                    n_ssm=1,            # Use a single ssm for all joint features
                    final_act='elu'     # Test different activation functions.
                )
            )

        self.temporal = nn.Sequential(*temporal_layers)     # (B, L, O)
        self.aggregator = lambda x: torch.mean(x, dim=1)    # (B, O)
        

class MultibranchAQA(nn.Module):
    def __init__(
        self,
        joint_count,
        joint_dim,
        joint_expansion,
        output_dim,
        layers_count,
        dropout
    ): 
        super().__init__()

        self.joint_count = joint_count
        self.joint_dim = joint_dim
        self.joint_expansion = joint_expansion
        self.output_dim = output_dim

        # One indipendent branch for each joint
        self.branches = nn.ModuleList([
            SingleJointTemporalBlock(
                input_dim=joint_dim,
                output_dim=joint_expansion,
                layers_count=layers_count,
                aggregator='mean',
                dropout=dropout
            ) for _ in range(joint_count)
        ])

        # =============================================
        # TODO: GNN over joints
        # self.gnn = SpatialGNN(...)
        # =============================================

        # NOTE: Allow different aggregators
        # NOTE: We could train a scorer for each joint, to show quality of each limb

        # How to aggregate joints output
        self.final_aggregator = lambda xs: torch.cat(xs, dim=1)

        # Final classifier/scorer
        aggregated_dim = joint_expansion * joint_count
        self.discriminator = nn.Sequential(
            nn.Linear(aggregated_dim, aggregated_dim),
            nn.ELU(),
            nn.Linear(aggregated_dim, output_dim)
        )

    def forward(self, x): # (B, J, L, F), one for joint
        assert(len(x.shape[1]) == self.joint_count)

        # Process each branch indipendently 
        jxs = torch.split(x, dim=1, split_size_or_sections=1)                           # J * (B, L, F)
        jxs = [ branch(joint.squeeze()) for branch, joint in zip(self.branches, jxs) ]  # J * (B, O)

        # TODO: Propagate spatially
        # ...

        x = self.final_aggregator(jxs)  # (B, JO)
        y = self.discriminator(x)       # (B, R)
        return y

