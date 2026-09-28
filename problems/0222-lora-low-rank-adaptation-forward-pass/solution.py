import torch

def lora_forward(
    x: torch.Tensor,
    W: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    alpha: float = 1.0
) -> torch.Tensor:
    """
    Compute the LoRA forward pass using PyTorch.
    
    Args:
        x: Input tensor (batch_size x in_features)
        W: Frozen pretrained weights (in_features x out_features)
        A: LoRA matrix A (rank x out_features)
        B: LoRA matrix B (in_features x rank)
        alpha: LoRA scaling factor
        
    Returns:
        Output tensor (batch_size x out_features)
    """
    rank = A.shape[0]
    base_output = torch.einsum("bi,ik->bk", x, W)
    lora_matrix = B @ A
    lora_output = torch.einsum("bi,ik->bk", x, lora_matrix)
    return base_output + (alpha / rank) * lora_output