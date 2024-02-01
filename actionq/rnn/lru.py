"""\
Implementation of Linear Recurrent Unit (LRU) from:
    https://arxiv.org/pdf/2303.06349.pdf
"""

import torch
import torch.nn as nn
import einops as ein
import math

# NOTE: Should use custom kernels
from actionq.utils.accel import associative_scan, binary_operator_diag

class LRU(nn.Module):
    """ Implementation of a Linear Recurrent Unit (LRU)
        https://arxiv.org/pdf/2303.06349.pdf
    """

    def __init__(
        self,
        state_dim,                  # The state dimension is the same as the input dimension and output dimension
        r_min=0.4,                  # Min. radius in the complex plane
        r_max=0.9,                  # Max. radius in the complex plane
        phase_max=math.pi * 2       # Phase in the form of [0, phase_max]
    ):
        super().__init__()

        self.state_dim = state_dim
        self.state = torch.complex(torch.zeros(state_dim), torch.zeros(state_dim))

        # Input to output, skip connection, implemented in the block
        # self.D = nn.Parameter(torch.randn([state_dim, state_dim]) / math.sqrt(state_dim))

        # Diagonal state matrix parameters
        u1 = torch.rand(state_dim)
        self.nu_log = nn.Parameter(torch.log(-0.5 * torch.log(u1 * (r_max + r_min) * (r_max - r_min) + r_min**2)))
        u2 = torch.rand(state_dim)
        self.theta_log = nn.Parameter(torch.log(phase_max * u2))

        # Diagonal state matrix and normalization factor
        Lambda_mod = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = nn.Parameter(torch.log(torch.sqrt(torch.ones_like(Lambda_mod) - torch.square(Lambda_mod))))

        # Input to state matrix
        B_re = torch.randn([state_dim, state_dim]) / math.sqrt(2 * state_dim)
        B_im = torch.randn([state_dim, state_dim]) / math.sqrt(2 * state_dim)
        self.B = nn.Parameter(torch.complex(B_re, B_im))

        # State to output matrix
        C_re = torch.randn([state_dim, state_dim]) / math.sqrt(state_dim)
        C_im = torch.randn([state_dim, state_dim]) / math.sqrt(state_dim)
        self.C = nn.Parameter(torch.complex(C_re, C_im))

    def forward(self, x):  # (B, L, F)
        self.state = self.state.to(self.B.device)

        # Istantiate diagonal state matrix
        L_mod = torch.exp(-torch.exp(self.nu_log))
        L_re = L_mod * torch.cos(torch.exp(self.theta_log))
        L_im = L_mod * torch.sin(torch.exp(self.theta_log))
        L_diag = torch.complex(L_re, L_im).to(self.B.device)

        # Istantiate normalization factor
        G_norm = torch.exp(self.gamma_log).unsqueeze(-1).to(self.B.device)
        B_norm = self.B * G_norm

        ## NOTE: this section requires optimization using parallel scan, it is unusable like this
        #result = torch.empty_like(x, device=self.B.device)
        #for i, batch in enumerate(x):
        #
        #    result_seq = torch.empty(x.shape[1], self.state_dim)
        #    for j, step in enumerate(batch):
        #        self.state = (L_diag * self.state + G_norm * self.B @ step.to(dtype=self.B.dtype))
        #        out_step = (self.C @ self.state).real  # + self.D @ step
        #        result_seq[j] = out_step
        #
        #    self.state = torch.complex(torch.zeros_like(self.state.real), torch.zeros_like(self.state.real))
        #    result[i] = result_seq

        L_elems = L_diag.tile(x.shape[1], 1)
        B_elems = x.to(B_norm.dtype) @ B_norm.T

        inner_state_fn = lambda B_seq: associative_scan(binary_operator_diag, (L_elems, B_seq))[1]
        inner_states = torch.vmap(inner_state_fn)(B_elems)
        return (inner_states @ self.C.T).real


class LRULayer(nn.Module):
    """ Wrapper for an LRU. Adds skip connection, normalization and stacking.
    """

    def __init__(
        self,
        state_dim,
        dropout,
        **lru_args
    ):
        super().__init__()

        # NOTE: Uses typical Norm-RNN-Activation-Dropout layout
        self.layer = nn.Sequential(
            nn.LayerNorm(state_dim),
            LRU(state_dim, **lru_args),
            nn.GELU(),  # Non-linearity,
            nn.Dropout(p=dropout)
        )

    def forward(self, x):  
        residual = x
        y = self.layer(x) + residual # (B, L, F)
        return y


class LRUModel(nn.Module):
    """Simple LRU module for temporal skeleton data
    """

    def __init__(
        self,
        joint_count,
        joint_features,
        joint_expansion,
        output_dim,
        layers_count,
        dropout
    ):
        super().__init__()

        self.initial_encoder = nn.Sequential(
            nn.Linear(joint_features, joint_expansion)
        )

        # NOTE: Use one aggregator for all joint time series
        self.temporal_layers = nn.ModuleList()
        for _ in range(layers_count):
            self.temporal_layers.append(
                LRULayer(
                    state_dim=joint_expansion,
                    dropout=dropout
                )
            )

        # How to aggregate temporal output (B, L, F) -> (B, F)
        # TODO: Try different aggregators
        self.temporal_aggregator = lambda x: torch.mean(x, 1)

        # From joint data obtains exercise score
        self.regressor = nn.Sequential(
            nn.Linear(joint_expansion * joint_count, joint_expansion * joint_count),
            nn.GELU(),
            nn.Linear(joint_expansion * joint_count, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """ Input x shape: (B, L, J, F)
        """

        B, L, J, F = x.shape

        x = self.initial_encoder(x)

        # Process temporal sequence to (BJ, F)
        x = ein.rearrange(x, 'B L J F -> (B J) L F')
        for temporal_layer in self.temporal_layers:
            x = temporal_layer(x)

        # Concat joint features
        x = self.temporal_aggregator(x)
        x = ein.rearrange(x, '(B J) K -> B (J K)', B=B, J=J)
        y = self.regressor(x)
        
        return y
