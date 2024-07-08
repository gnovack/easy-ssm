import torch
from dataclasses import dataclass
from util import gbt, hippo_matrix


@dataclass
class LSSLConfig:
    """Configuration for Linear State-Space Layer as
    introduced in https://arxiv.org/abs/2110.13985"""

    # The size of each state vector. Denoted by 'N' in the paper.
    state_size: int
    
    # The number of hidden dimensions (i.e. distinct copies of the state space model).
    # Denoted by 'H' in the paper.
    hidden_size: int

    # The number of output channels. Denoted by 'M' in the paper.
    num_channels: int

    # The number of stacked LSSL layers in the model.
    num_layers: int

    dropout: float = 0.2

    initial_dt: float = 0.01

    train_A: bool = False
    train_dt: bool = False


class LSSL(torch.nn.Module):
    def __init__(self, config: LSSLConfig):
        super(LSSL, self).__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.num_channels = config.num_channels
        self.state_size = config.state_size
        
        A, B = hippo_matrix(self.state_size)

        # Continuous-time parameters
        self.A = torch.nn.Parameter(A)
        self.B = torch.nn.Parameter(B)

        # TODO - initialize dt log-uniformly between min_dt and max_dt
        self.dt = torch.nn.Parameter(torch.ones(self.hidden_size) * config.initial_dt)

        self.A.requires_grad = config.train_A
        self.dt.requires_grad = config.train_dt
        self.cache_convolutional_kernel = (config.train_A == False) and (config.train_dt == False)

        self.C = torch.nn.Parameter(torch.randn(self.hidden_size, self.num_channels, self.state_size))
        self.D = torch.nn.Parameter(torch.randn(self.hidden_size, self.num_channels))
        self.output_proj = torch.nn.Parameter(torch.randn(self.hidden_size * self.num_channels, self.hidden_size))


    def forward(self, input, mode='recurrent'):

        # Discretize the continuous-time model using the generalized bilinear transform
        A_discrete, B_discrete = gbt(self.A, self.B, self.dt)

        batch_size, sequence_length = input.shape[0], input.shape[1]

        # Initialize hidden state to zeros
        x = torch.zeros(batch_size, self.hidden_size, self.state_size)

        if mode == 'recurrent':

            # List to store the output at each time step
            outputs = []

            for i in range(sequence_length):

                # Select the input at index i in the sequence
                current_input = input[:, i, :]
                
                # Update the current state
                Ax = (A_discrete @ x.unsqueeze(-1)).reshape(*x.shape)
                Bu = torch.einsum('hn, bh -> bhn', B_discrete, current_input)
                x = Ax + Bu

                # Compute the output at the current time step
                Cx = torch.einsum('hmn, bhn -> bhm', self.C, x)
                Du = torch.einsum('hm, bh -> bhm', self.D, current_input)
                y = Cx + Du
                
                outputs.append(y)
            
            y = torch.stack(outputs, dim=1)

        if mode == 'convolutional':
            K = self.compute_convolution_kernel(sequence_length, A_discrete, B_discrete, self.C)

            conv_input = torch.nn.functional.pad(input.transpose(1, -1), (sequence_length-1, 0))

            y = torch.conv1d(
                input=conv_input,   # [B, H, L]
                weight=K,           # [ H * M, 1, L]
                groups=self.hidden_size
            )
            y = y.transpose(1, 2).reshape(batch_size, sequence_length, self.hidden_size, self.num_channels)
            
            Du = torch.einsum('hm, blh -> blhm', self.D, input)
            y += Du

        # Output projection
        y = torch.nn.functional.gelu(y)
        output = y.view(batch_size, sequence_length, -1) @ self.output_proj
        return output
            

    def compute_convolution_kernel(self, sequence_length, A_discrete, B_discrete, C):
        K = [B_discrete.unsqueeze(-1)]
        for _ in range(1, sequence_length):
            K.append(A_discrete @ K[-1])

        K.reverse()
        K = torch.stack(K, dim=-1).squeeze(-2)
        return (C @ K).view(self.hidden_size * self.num_channels, 1, -1)


class LSSLModel(torch.nn.Module):

    def __init__(self, config: LSSLConfig):
        super(LSSLModel, self).__init__()
        self.blocks = torch.nn.ModuleList([LSSL(config) for _ in range(config.num_layers)])
        self.norm = torch.nn.LayerNorm(config.hidden_size)
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, u, mode='recurrent'):
        for block in self.blocks:
            residual = u
            u = self.norm(u)
            u = block(u, mode)
            u = self.dropout(u) + residual
        return u


class LSSLModelForSequenceClassification(torch.nn.Module):

    def __init__(self, config: LSSLConfig, num_classes: int):
        super(LSSLModelForSequenceClassification, self).__init__()

        self.in_proj = torch.nn.Linear(1, config.hidden_size)
        self.lssl = LSSLModel(config)
        self.fc = torch.nn.Linear(config.hidden_size, num_classes)

    def forward(self, u, mode='recurrent'):
        u = self.in_proj(u)
        u = self.lssl(u, mode)
        u = self.fc(u)
        return u
