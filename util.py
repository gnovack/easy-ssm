import torch
from typing import Tuple


def gbt(A: torch.Tensor, B: torch.Tensor, dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Discretize a continuous-time system using the generalized bilinear transformation.
    
    From Gu, Albert, et al. (https://arxiv.org/abs/2206.11893), the generalized
    bilinear transformation can be defined as:

    ```
    A_discrete = (I - Δ/2 * A)⁻¹(I + Δ/2 * A)
    B_discrete = (I - Δ/2 * A)⁻¹ ⋅ ΔB
    ```
    """
    dt = dt.view(-1, 1, 1)
    I = torch.eye(A.shape[-1], device=A.device)

    dtA = (dt / 2.0) * A

    A_discrete = torch.inverse(I - dtA) @ (I + dtA)
    B_discrete = torch.inverse(I - dtA) @ (B * dt)
    return A_discrete, B_discrete.squeeze(-1)


def hippo_matrix(state_size):
    """Computes the HiPPO-LegS Matrix of the given state size.

    From Gu, Albert, et al. (https://arxiv.org/abs/2008.07669), the 
    HiPPO-LegS Matrix is given by:

    ```
               ⎧ (2n + 1)¹ᐟ²(2k + 1)¹ᐟ² if n > k
    A[n,k] = - ⎨ n + 1                 if n = k       
               ⎩ 0                     if n < k

    B[n]   = (2n + 1)¹ᐟ²   
    ```
    """
    A = torch.diag(torch.arange(1, state_size+1, dtype=torch.float))

    def lower_triangle_elements(n, k): 
        return (2*n + 1)**0.5 * (2*k + 1)**0.5

    for i in range(1, state_size):
        for j in range(i):
            A[i, j] = lower_triangle_elements(i, j)
    
    B = A[:, 0].reshape(-1, 1)
    return -1*A, B
