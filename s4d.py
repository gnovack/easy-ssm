import math
import torch
from dataclasses import dataclass
from util import gbt


@dataclass
class S4DConfig:
    state_size: int
    hidden_size: int
    num_layers: int
    dropout: float = 0.2
    max_dt: float = 0.01
    min_dt: float = 0.001


class S4D(torch.nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, config: S4DConfig):
        super(S4D, self).__init__()

        self.hidden_size = config.hidden_size
        self.state_size = config.state_size

        # Initialize Δt log-uniformly between min_dt and max_dt
        log_dt_min, log_dt_max = math.log(config.min_dt), math.log(config.max_dt) 
        log_dt = torch.rand(self.hidden_size) * (log_dt_max - log_dt_min) + log_dt_min
        self.log_dt = torch.nn.Parameter(log_dt)

        C = torch.randn(self.hidden_size, self.state_size, dtype=torch.cfloat)
        self.C = torch.nn.Parameter(torch.view_as_real(C))
        self.D = torch.nn.Parameter(torch.randn(self.hidden_size))

        # Initialize A using the S4D-Lin Method
        A = torch.tensor([0.5 + (math.pi * n * 1j) for n in range(self.state_size)]).repeat(self.hidden_size, 1)
        
        # Initialize B to 1
        B = torch.ones(self.state_size, 1, dtype=torch.cfloat)

        self.log_A_real = torch.nn.Parameter(torch.log(A.real))
        self.A_imag = torch.nn.Parameter(A.imag)
        self.B = torch.nn.Parameter(B)

        # Initialize output projection matrices W₁ and W₂
        self.output_proj_1 = torch.nn.Parameter(torch.randn(self.hidden_size, self.hidden_size))
        self.output_proj_2 = torch.nn.Parameter(torch.randn(self.hidden_size, self.hidden_size))

        self.activation = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, input, mode='recurrent'):

        batch_size, sequence_length = input.shape[0], input.shape[1]

        dt = torch.exp(self.log_dt)

        # Discretize the continuous-time model using the generalized bilinear transform
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        A_discrete, B_discrete = gbt(A=torch.diag_embed(A), B=self.B, dt=dt)
        C = torch.view_as_complex(self.C)

        if mode == 'recurrent':
            outputs = []
            x = torch.zeros(batch_size, self.hidden_size, self.state_size, dtype=torch.cfloat)

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
            K = self.compute_convolution_kernel(sequence_length, torch.diagonal(A_discrete, dim1=1, dim2=2), B_discrete, C)

            conv_input = torch.nn.functional.pad(input.transpose(1, -1), (sequence_length-1, 0))

            y = torch.conv1d(
                input=conv_input,   
                weight=K,           
                groups=self.hidden_size
            )
            y = y.transpose(1,2).reshape(batch_size, sequence_length, self.hidden_size)
            
            Du = torch.einsum('h, blh -> blh', self.D, input)
            y += Du

        output = torch.nn.functional.glu(
            torch.stack([y @ self.output_proj_1, y @ self.output_proj_2], dim=-1)
        ).squeeze(-1)
        return output
    
    def compute_convolution_kernel(self, sequence_length, A_discrete, B_discrete, C):
        vandermonde_A = torch.linalg.vander(A_discrete, N=sequence_length)
        K = torch.einsum('hn, hnl -> hl', B_discrete * C, vandermonde_A).unsqueeze(1)
        return K.real.flip(-1)
    

class S4DModel(torch.nn.Module):

    def __init__(self, config: S4DConfig):
        super(S4DModel, self).__init__()
        self.blocks = torch.nn.ModuleList([S4D(config) for _ in range(config.num_layers)])
        self.norm = torch.nn.LayerNorm(config.hidden_size)
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, u, mode='recurrent'):
        for block in self.blocks:
            residual = u
            u = self.norm(u)
            u = block(u, mode)
            u = u + residual
        return u


class S4DModelForSequenceClassification(torch.nn.Module):

    def __init__(self, config: S4DConfig, vocab_size: int, num_classes: int):
        super(S4DModelForSequenceClassification, self).__init__()

        self.input_embeds = torch.nn.Embedding(vocab_size, config.hidden_size)
        self.s4d = S4DModel(config)
        self.fc = torch.nn.Linear(config.hidden_size, num_classes)

    def forward(self, u, mode='recurrent'):
        u = self.input_embeds(u)
        u = self.s4d(u, mode)
        u = self.fc(u)
        return u
