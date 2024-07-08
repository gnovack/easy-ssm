import math
import torch
from dataclasses import dataclass
from util import gbt


@dataclass
class H3Config:
    state_size: int
    hidden_size: int
    max_dt: float = 0.01
    min_dt: float = 0.001


class SSM(torch.nn.Module):
    """Generic SSM to support both Shift and Diag models."""

    def __init__(self, config: H3Config, init_A="s4d-lin"):
        super(SSM, self).__init__()

        self.hidden_size = config.hidden_size
        self.state_size = config.state_size

        # Initialize Δt log-uniformly between min_dt and max_dt
        log_dt_min, log_dt_max = math.log(config.min_dt), math.log(config.max_dt) 
        log_dt = torch.rand(self.hidden_size) * (log_dt_max - log_dt_min) + log_dt_min
        self.log_dt = torch.nn.Parameter(log_dt)

        self.D = torch.nn.Parameter(torch.randn(self.hidden_size))

        self.init_A = init_A
        if self.init_A == "s4d-lin":
            # Initialize A using the S4D-Lin Method
            A = torch.tensor([0.5 + (math.pi * n * 1j) for n in range(self.state_size // 2)]).repeat(self.hidden_size, 1)
            self.A_imag = torch.nn.Parameter(A.imag)
            A = torch.log(A.real)
            
            B = torch.randn(self.hidden_size, self.state_size // 2, dtype=torch.cfloat)
            B = torch.view_as_real(B)

            C = torch.randn(self.hidden_size, self.state_size // 2, dtype=torch.cfloat)
            C = torch.view_as_real(C)

        elif self.init_A == "shift":
            # Initialize A to a Shift matrix with 1s just below the main diagonal
            A = torch.diag(torch.ones(self.state_size-1), diagonal=-1)
            A.requires_grad = False

            B = torch.randn(self.hidden_size, self.state_size)
            C = torch.randn(self.hidden_size, self.state_size)
        
        self.A = torch.nn.Parameter(A)
        self.B = torch.nn.Parameter(B)
        self.C = torch.nn.Parameter(C)

    def forward(self, input, mode='recurrent'):

        batch_size, sequence_length = input.shape[0], input.shape[1]
        dt = torch.exp(self.log_dt)

        # Discretize the continuous-time model using the generalized bilinear transform
        if self.init_A == "s4d-lin":
            A = -torch.exp(self.A) + 1j * self.A_imag
            A_discrete, B_discrete = gbt(
                A=torch.diag_embed(A),
                B=torch.view_as_complex(self.B).unsqueeze(-1),
                dt=dt
            )
            C = torch.view_as_complex(self.C) * 2
        elif self.init_A == "shift":
            # No discretization is needed when using the fixed shift matrix
            A_discrete = self.A
            B_discrete = self.B
            C = self.C

        if mode == 'recurrent':
            outputs = []
            x = torch.zeros(batch_size, self.hidden_size, A_discrete.shape[1], dtype=A_discrete.dtype, device=input.device)

            for i in range(sequence_length):

                # Select the input at index i in the sequence
                current_input = input[:, i, :]

                 # Update the current state
                Ax = (A_discrete @ x.unsqueeze(-1)).reshape(*x.shape)
                Bu = torch.einsum('hn, bh -> bhn', B_discrete, current_input)
                x = Ax + Bu

                # Compute the output at the current time step
                Cx = torch.einsum('hn, bhn -> bh', C, x).real
                Du = torch.einsum('h, bh -> bh', self.D, current_input)
                
                y = Cx + Du
                outputs.append(y)
            y = torch.stack(outputs, dim=1)

        if mode == 'convolutional':

            if self.init_A == "s4d-lin":
                # Compute K using the Vandermonde matrix for diagonal A
                A_discrete = torch.diagonal(A_discrete, dim1=1, dim2=2)
                K = self.compute_convolution_kernel_vandermonde(sequence_length, A_discrete, B_discrete, C)
            else:
                # Compute K the old fashioned way for non-diagonal A
                K = self.compute_convolution_kernel(sequence_length, A_discrete, B_discrete, C)

            conv_input = torch.nn.functional.pad(input.transpose(1, -1), (sequence_length-1, 0))

            y = torch.conv1d(
                input=conv_input,   
                weight=K,           
                groups=self.hidden_size
            )
            y = y.transpose(1,2).reshape(batch_size, sequence_length, self.hidden_size)

            Du = torch.einsum('h, blh -> blh', self.D, input)
            y += Du

        return y

    def compute_convolution_kernel_vandermonde(self, sequence_length, A_discrete, B_discrete, C):
        vandermonde_A = torch.linalg.vander(A_discrete, N=sequence_length)
        K = torch.einsum('hn, hnl -> hl', B_discrete * C, vandermonde_A).unsqueeze(1)
        return K.real.flip(-1)

    def compute_convolution_kernel(self, sequence_length, A_discrete, B_discrete, C):
        K = [B_discrete.unsqueeze(-1)]
        for _ in range(1, sequence_length):
            K.append(A_discrete @ K[-1])

        K.reverse()
        K = torch.stack(K, dim=-1).squeeze(-2)
        return (C.unsqueeze(1) @ K).view(self.hidden_size, 1, -1)


class H3(torch.nn.Module):

    def __init__(self, config: H3Config):
        super(H3, self).__init__()

        self.state_size = config.state_size
        self.hidden_size = config.hidden_size

        self.ssm_shift = SSM(config, init_A="shift")
        self.ssm_diag = SSM(config, init_A="s4d-lin")

        self.q_proj = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = torch.nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, input, mode='recurrent'):
        q = self.q_proj(input)
        k = self.k_proj(input)
        v = self.v_proj(input)

        k = self.ssm_shift(k, mode)
        kv = k * v
        kv = self.ssm_diag(kv, mode)

        qkv = (q * kv)
        return self.out_proj(qkv)
