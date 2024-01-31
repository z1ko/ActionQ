"""\
Implementation of Linear Recurrent Unit (LRU) from:
    https://arxiv.org/pdf/2303.06349.pdf
"""

import torch
import torch.nn as nn
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


class LRUAggregator(nn.Module):
    """ Wrapper for an LRU. Adds skip connection, normalization,
    feature projections and stacking.
    """

    def __init__(
        self,
        input_dim,
        state_dim,
        output_dim,
        layers_count,
        dropout,
        aggregator='mean',
        **lru_args
    ):
        super().__init__()

        self.encoder = nn.Linear(input_dim, state_dim)
        self.decoder = nn.Linear(state_dim, output_dim)

        # NOTE: Uses typical Norm-RNN-Activation-Dropout layout
        self.layers = nn.ModuleList()
        for _ in range(layers_count):
            self.layers.append(
                nn.Sequential(
                    nn.LayerNorm(state_dim),
                    LRU(state_dim, **lru_args),
                    nn.ELU(),  # Non-linearity,
                    nn.Dropout(p=dropout)
                )
            )

        # How to aggregate temporal output (B, L, F) -> (B, F)
        # TODO: Try different aggregators
        if aggregator == 'mean':
            self.agg = lambda x: torch.mean(x, 1)
        else:
            raise ValueError(f'Aggregator {aggregator} not valid for LRUBlock')

    def forward(self, x):  # (B, L, F)

        x = self.encoder(x)

        for layer in self.layers:
            residual = x
            x = layer(x) + residual

        x = self.agg(x)
        y = self.decoder(x)

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

        self.state_dim = joint_expansion * joint_count
        self.encoder = nn.Linear(joint_features, joint_expansion)

        # NOTE: for now use only one for all joint data
        self.temporal = LRUAggregator(
            input_dim=self.state_dim,
            state_dim=self.state_dim,
            output_dim=self.state_dim,
            layers_count=layers_count,
            dropout=dropout
        )

        # From joint data obtains exercise score
        self.regressor = nn.Sequential(
            nn.Linear(self.state_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """ Input x shape: (B, L, J, F)
        """

        B, L, J, F = x.shape

        # Change features dimensions (B, L, J, K)
        x = self.encoder(x)

        # Compact last dimensions (B, L, JK)
        x = x.view(B, L, J * x.shape[-1])

        # Process temporal sequence (B, JK)
        x = self.temporal(x)

        y = self.regressor(x)
        return y
