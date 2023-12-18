
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class S4D(nn.Module):
    """
        Structured State Space Model wih Diagonal HiPPO matrix
        Based on:
            S4D https://arxiv.org/abs/2206.11893
            DSS https://arxiv.org/abs/2203.14343
    """

    def __init__(self, n_channels, d_state, dt_min=0.001, dt_max=0.1):
        """
            Arguments:
                n_channels: number of indipendent SSM, each one is assigned to a single input channel
                d_state: dimension of the hidden state
                dt_min, dt_max: range of the discretization
        """
        super().__init__()
        self._init_parameters(n_channels, d_state, dt_min, dt_max)

    def _init_parameters(self, n_channels, d_state, dt_min, dt_max):
        """
            Initialize matrices of the state model
        """

        # Geometric uniform timescale
        # NOTE: Controls how much the model is biased towards recent interactions
        self.dt_log = torch.rand() * (torch.log(dt_max) - torch.log(dt_min)) + torch.log(dt_min)

        # The matrices are have symmetrical entries that are conjugate, no need to store the complete
        # state size, only half of it is needed
        d_state_complex = d_state // 2

        # S4D-Lin initialization (contains only diagonal elements)
        # TODO: Allow different initializations
        A = -0.5 + 1j * torch.pi * torch.arange(d_state_complex)
        A = einops.repeat(A, 'd -> c d', c=n_channels)
        self.log_A_real = A.real
        self.A_imag = A.imag

        # Variance preserving initialization
        self.C = torch.randn(n_channels, d_state_complex, dtype=torch.cfloat)
        # Random initialization
        self.B = torch.ones((n_channels, d_state_complex), dtype=torch.cfloat)

    def _discretize(self, dt, A, B):
        # TODO: Allow different discretization methods
        dA = (1 + dt * A / 2) / (1 - dt * A / 2)
        dB = dt * B / (1 - dt * A / 2)
        return dA, dB

    def kernel(self, L):
        """
            Create the SSM's convolution kernel
        """

        # Materialize state matrix
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag

        # Discretized matrices
        dt = torch.exp(self.dt_log)
        dA, dB = self._discretize(dt, A, self.B)

        # Compute kernel using Vandermonde matrix multiplication
        # TODO: Optimize
        return 2 * ((self.B * self.C) @ (dA[:, None] ** torch.arange(L))).real

    def forward(self, U):
        """
            Process entire input using convolutional mode
            Arguments:
                U: input tensor of shape (batch, channels, time)
        """

        # Create kernel on the input lenght
        L = U.size(-1)
        K = self.kernel(L)

        # Convolve y = u * K using FFT
        conv_size = 2 * L
        K_freq, U_freq = torch.fft.rfft(K, n=conv_size), torch.fft.rfft(U, n=conv_size)
        return torch.fft.irfft(K_freq * U_freq, n=conv_size)[..., :L]
