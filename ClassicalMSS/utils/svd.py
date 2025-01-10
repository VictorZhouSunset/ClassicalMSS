# Copyright (c) Victor Zhou
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Structure of the project is learned from https://github.com/facebookresearch/demucs

import random
import torch

def power_iteration_old(m, num_iters=1, bs=1):
    """
    This is the power method. batch size is used to try multiple starting point in parallel.
    This is the original implementation from the Demucs repo, I wonder why they don't use the Rayleigh quotient way
    to compute the largest eigenvalue, this way may save some time (compared to the torch.svd_lowrank) but with a big loss in accuracy
    """
    assert m.dim() == 2, "Input matrix must be 2D"
    assert m.shape[0] == m.shape[1], "Input matrix must be square"
    dim = m.shape[0]
    b = torch.randn(dim, bs, device=m.device, dtype=m.dtype) # torch.randn() generates random numbers from a normal (Gaussian) distribution

    for _ in range(num_iters):
        n = m.mm(b)
        norm = n.norm(dim=0, keepdim=True) # Compute the norm of each column since different columns are different batches
        b = n / (norm + 1e-10) # Avoid division by zero

    return norm.mean() # Why not .max()?

def power_iteration(m, num_iters=1, bs=1):
    """
    This is the power method. batch size is used to try multiple starting point in parallel.
    This method uses the Rayleigh quotient to compute the largest singular value.
    This way it computes faster and more accurately than the torch.svd_lowrank when num_iters is big.
    Still should use the torch.svd_lowrank for smaller num_iters under 10 or when the batch size (bs) is small.
    For detailed comparison, see the ClassicalMSS/tests/test_svd.py file.
    """
    assert m.dim() == 2, "Input matrix must be 2D"
    assert m.shape[0] == m.shape[1], "Input matrix must be square"
    dim = m.shape[0]
    b = torch.randn(dim, bs, device=m.device, dtype=m.dtype) # torch.randn() generates random numbers from a normal (Gaussian) distribution
    b = b / b.norm(dim=0, keepdim=True) # Normalize the starting point

    for _ in range(num_iters):
        n = m.mm(b)
        norm = n.norm(dim=0, keepdim=True) # Compute the norm of each column since different columns are different batches
        b = n / (norm + 1e-10) # Avoid division by zero
    
    mb = m.mm(b)  # [dim, bs]
    estimate = torch.sum(b * mb, dim=0)  # Element-wise multiplication and then sum
    
    return estimate.max()

# Unsing a dedicated random number generator (RNG) with a fixed seed (here 1234)
# This is for the extendability to later distributed training
penalty_rng = random.Random(1234)

def svd_penalty(
    model,
    min_size=0.1,
    dim=1,
    num_iters=2,
    power_method=False,
    convtr=True,
    proba=1,
    conv_only=False,
    exact=False,
    bs=1
):
    """
    Penalty on the largest singular value (through the SVD: singular value decomposition)
    of the weight matrices of a model.
    Args:
        - model: model to penalize
        - min_size: minimum size in MB of a layer to penalize.
        - dim: projection dimension for the svd_lowrank. Higher is better but slower.
        - num_iters: number of iterations in the algorithm used by svd_lowrank.
        - power_method: use power method instead of lowrank SVD, the Demucs repo thinks
            it is both slower and less stable.
        - convtr: when True, differentiate between Conv and Transposed Conv by swapping
            the input and output channels of the Transposed Conv layers.
        - proba: probability to apply the penalty.
        - conv_only: only apply to conv and conv transposed, not LSTM
            (might not be reliable for other models than Demucs).
        - exact: use exact SVD (slow but useful at validation).
        - bs: batch_size for power method.
    """
    total = 0
    if penalty_rng.random() > proba:
        return 0. # Float literal instead of an integer
    
    for m in model.modules():
        for name, param in m.named_parameters(recurse=False):
            # The outer loop goes through all the modules (and those are nested) in the model
            # So no recursion is needed
            if param.numel() / 2**18 < min_size:
                continue # Skip layers that are too small to penalize
            if convtr:
                """
                The ConvTranspose layers are used in the upsampling part of the models.
                For example in the (1d) in_channels = 4, out_channels = 2, kernel_size = 2 case,
                Every kernel window transforms the 4 (in_channels) * 1 (sample) input to 2 (out_channels) * 2 (samples) output.
                In this case, it may be better to multiply the out_channels with the kernel_size rather than the in_channels.
                """
                if isinstance(m, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d)):
                    # For ConvTranspose1d, the param is of shape (out_channels, in_channels, kernel_size)
                    # For ConvTranspose2d, the param is of shape (out_channels, in_channels, kernel_height, kernel_width)
                    if param.dim() in [3, 4]:
                        param = param.transpose(0, 1).contiguous()
            if param.dim() == 3:
                # Reshape the tensor to 2D: first dimension stays the same,
                # the second dimension is the multiplication of all the other dimensions
                param = param.view(len(param), -1) 
            elif param.dim() == 4:
                param = param.view(len(param), -1)
            elif param.dim() == 1: # The bias terms / BatchNorm layers, etc.
                continue
            elif conv_only:
                continue
            assert param.dim() == 2, f"Unexpected dimension {param.dim()} for {name} with shape {param.shape}"
            
            # Now we have a 2D tensor, the following is for the SVD part
            if exact:
                # torch.svd decomposes a matrix into U Σ V^T where Σ contains the singular values
                # We only need the singular values (Σ), so we set compute_uv=False
                # Although the U and V are not calculated, the output is still a tuple of 3 tensors (U-empty, Σ, V-empty)
                # So select the second tensor (Σ) (indexed 1)
                # The singular values are non-negative, but squaring them may provide
                # smoother gradients and provide easier implementation of other svd approximations, etc.
                # Pick the maximum
                estimate = torch.svd(param, compute_uv=False)[1].pow(2).max()
            elif power_method:
                a, b = param.shape
                # AA^T and A^TA will have the same non-zero eigenvalues (as square matrices)
                # Here we always choose the smaller matrix size for efficiency
                if a < b: # mm means matrix multiplication
                    n = param.mm(param.t()) # computes param * param.t()
                else:
                    n = param.t().mm(param) # computes param.t() * param
                estimate = power_iteration(n, num_iters, bs)
            else:
                # torch.svd_lowrank only returns the first (largest) q singular values
                estimate = torch.svd_lowrank(param, q=dim, niter=num_iters)[1][0].pow(2)
            
            total += estimate
    
    return total / proba # To maintain the same expected value of the penalty

            


