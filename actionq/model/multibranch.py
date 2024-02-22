import torch
import torch.nn as nn
import einops as ein

from typing import List
from actionq.rnn.lru import LRULayer

class SingleJointTemporalBlock(nn.Module):
    def __init__(
        self,
        state_dim,
        layers_count,
        dropout,
        **kwargs
    ):
        super().__init__()

        self.temporal_layers = nn.ModuleList([
            LRULayer(
                state_dim=state_dim,
                activation='gelu',
                dropout=dropout,
                **kwargs
            ) for _ in range(layers_count)
        ])
        
    def forward(self, x): # (B, L, F) 
        for layer in self.temporal_layers:
            x = layer(x)
        return x

    def forward_with_state(self, x, state):
        for i, layer in enumerate(self.temporal_layers):
            x, state[i] = layer.forward_with_state(x, state[i])
        return x, state


class Parallel(nn.ModuleList):
    def __init__(self, modules: List[nn.Module], split_dim: int):
        super().__init__(modules)
        self.split_dim = split_dim

    # TODO: Force torch to use parallelism
    def forward(self, x: torch.Tensor) -> torch.Tensor: # (B, L, J, F)
        jxs = torch.split(x, split_size_or_sections=1, dim=self.split_dim)
        results = [ module(x.squeeze()) for module, x in zip(self, jxs) ]
        return torch.stack(results, dim=self.split_dim)

    def forward_with_state(self, x, state):
        jxs = torch.split(x, split_size_or_sections=1, dim=self.split_dim)
        results = [ module.forward_with_state(jxs[i].squeeze(), state[i]) for i, module in enumerate(self) ]
        return torch.stack(results, dim=self.split_dim)


class SkeletonGAT(nn.Module):
    def __init__(self, input_dim, joint_count, dropout):
        super().__init__()

        self.keys = nn.Linear(input_dim, input_dim)
        self.queries = nn.Linear(input_dim, input_dim)
        self.values = nn.Linear(input_dim, input_dim)

        self.norm = nn.LayerNorm(input_dim)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout)

        # Semantic features for each joint
        # TODO: use more significant representation combining symmetry and body region
        # self.I = torch.zeros((joint_count, joint_count))
        # for i in range(self.I.shape[0]):
        #     self.I[i,i] = 1.0

    def self_attention(self, x):                    # (B, L, J, F)
        K = self.keys(x)                            # (B, L, J, K)
        Q = self.queries(x)                         # (B, L, J, Q)
        V = self.values(x)                          # (B, L, J, V)

        s = torch.matmul(Q, K.transpose(-2, -1))    # (B, L, J, J)
        w = nn.functional.softmax(s, dim=-1)        # (B, L, J, J)

        return torch.matmul(w, V)                   # (B, L, J, R)

    def forward(self, x): # (B, L, J, F)
        res = x

        x = self.norm(x)
        x = self.self_attention(x) # (B, L, J, R)
        x = self.act(x)
        x = self.dropout(x)

        return x + res


class MultibranchModel(nn.Module):
    __constants__ = ['branches']

    def __init__(
        self,
        joint_count,
        joint_features,
        joint_expansion,
        output_dim,
        temporal_layers_count,
        dropout,
        **kwargs
    ): 
        super().__init__()

        self.joint_count = joint_count
        self.joint_dim = joint_features
        self.joint_expansion = joint_expansion
        self.output_dim = output_dim

        self.initial = nn.Sequential(
            nn.Linear(joint_features, joint_expansion),
            nn.LeakyReLU(),
        )

        # One indipendent branch for each joint
        self.temporal_model = Parallel(
            [
                SingleJointTemporalBlock(
                    state_dim=joint_expansion,
                    layers_count=temporal_layers_count,
                    dropout=dropout,
                    **kwargs
                ) for _ in range(joint_count)
            ], 
            split_dim=2
        )

        self.spatial_model = SkeletonGAT(
            input_dim=joint_expansion,
            joint_count=joint_count,
            dropout=dropout
        )

        # NOTE: Allow different aggregators
        # NOTE: We could train a scorer for each joint, to show quality of each limb

        #self.final = nn.Sequential(
        #    nn.Linear(128 * joint_count, 128),
        #    nn.LeakyReLU(),
        #    nn.Linear(128, output_dim),
        #    nn.Sigmoid()
        #)

        self.final = nn.Sequential(
            nn.Linear(joint_expansion, joint_expansion),
            nn.LeakyReLU(),
            nn.Linear(joint_expansion, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x): # (B, L, J, F), one for joint
        assert(x.shape[2] == self.joint_count)
        B, L, J, F = x.shape

        x = self.initial(x)

        x = self.temporal_model(x)
        x = self.spatial_model(x)

        # Spatial max-pooling
        x = ein.rearrange(x, 'B L J F -> (B L) F J')
        x = nn.functional.max_pool1d(x, kernel_size=J).squeeze()

        # Temporal mean-pooling
        x = ein.rearrange(x, '(B L) F -> B F L', B=B, L=L)
        x = torch.mean(x, dim=-1) # (B F)

        y = self.final(x)
        return y

    def forward_with_state(self, x, state):
        """ Input x shape (J, F), state shape (T, J, S)
        """

        x = self.initial(x)
        x = self.temporal_model.forward_with_state(x, state)
        x = self.condenser(x)

        x = ein.rearrange(x, 'J F -> (J F)')
        y = self.final(x).squeeze(-1)

        return y

