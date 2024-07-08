import torch
from dataclasses import dataclass


@dataclass
class MambaConfig:
    # 𝑁 – The SSM state vector size
    state_size: int
    
    # 𝐷 - The model dimension
    hidden_size: int
    
    # 𝐸 – Model dimension expansion factor 
    expansion_factor: int

    # 𝑅 - The time step rank
    dt_rank: int

    # Kernel size used in the 1D convolution
    conv_kernel_size: int


class MambaSSM(torch.nn.Module):

    def __init__(self, config: MambaConfig):
        super(MambaSSM, self).__init__()

        self.hidden_size = config.hidden_size * config.expansion_factor
        self.state_size = config.state_size

        self.dt_proj_in = torch.nn.Linear(self.hidden_size, config.dt_rank, bias=False)
        self.dt_proj_out = torch.nn.Linear(config.dt_rank, self.hidden_size, bias=True)

        self.B_proj = torch.nn.Linear(self.hidden_size, self.state_size, bias=False)
        self.C_proj = torch.nn.Linear(self.hidden_size, self.state_size, bias=False)
        self.D = torch.nn.Parameter(torch.randn(self.hidden_size))

        A = torch.arange(1, self.state_size + 1).repeat(self.hidden_size, 1).float()
        self.A = torch.nn.Parameter(torch.log(A))

    def forward(self, input):

        batch_size, sequence_length = input.shape[0], input.shape[1]

        A = -torch.exp(self.A)
        B = self.B_proj(input)
        C = self.C_proj(input)
        
        dt = self.dt_proj_in(input)
        dt = self.dt_proj_out(dt)
        dt = torch.nn.functional.softplus(dt)

        A_discrete = torch.exp(
            A.view(1, 1, self.hidden_size, self.state_size) * 
            dt.view(batch_size, sequence_length, self.hidden_size, 1)
        )
        B_discrete = (
            B.view(batch_size, sequence_length, 1, self.state_size) * 
            dt.view(batch_size, sequence_length, self.hidden_size, 1)
        )

        outputs = []
        x = torch.zeros(batch_size, self.hidden_size, self.state_size)
        for i in range(sequence_length):

            # Select the input at index i in the sequence
            current_input = input[:, i, :]
            
            # Because A, B, and C are time-variant, we need to select the 
            # values for the current time step
            current_A_discrete = A_discrete[:, i, :, :]
            current_B_discrete = B_discrete[:, i, :, :]
            current_C = C[:, i, :]

            # Update the current state
            Ax = torch.einsum('bhn, bhn -> bhn', current_A_discrete, x)
            Bu = torch.einsum('bhn, bh -> bhn', current_B_discrete, current_input)
            x = Ax + Bu

            # Compute the output at the current time step
            Cx = torch.einsum('bnx, bhn -> bh', current_C.unsqueeze(-1), x)
            Du = torch.einsum('h, bh -> bh', self.D, current_input)
            
            y = Cx + Du
            outputs.append(y)
        y = torch.stack(outputs, dim=1)
        return y


class Mamba(torch.nn.Module):

    def __init__(self, config: MambaConfig):
        super(Mamba, self).__init__()

        self.state_size = config.state_size
        self.hidden_size = config.hidden_size * config.expansion_factor
        self.conv_kernel_size = config.conv_kernel_size

        self.conv = torch.nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=self.conv_kernel_size,
            padding=self.conv_kernel_size - 1,
            groups=self.hidden_size
        )
        self.ssm = MambaSSM(config)

        self.q_proj = torch.nn.Linear(config.hidden_size, self.hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(config.hidden_size, self.hidden_size, bias=False)
        self.out_proj = torch.nn.Linear(self.hidden_size, config.hidden_size, bias=False)

    def forward(self, input):
        sequence_length = input.shape[1]

        q = self.q_proj(input)
        k = self.k_proj(input)

        conv_out = self.conv(k.transpose(1, 2))[..., :sequence_length]
        conv_out = torch.nn.functional.silu(conv_out)
        ssm_out = self.ssm(conv_out.transpose(1, 2))

        output = ssm_out * torch.nn.functional.silu(q)
        return self.out_proj(output)
