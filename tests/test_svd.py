# Copyright (c) Victor Zhou
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
from pathlib import Path
root = Path(__file__).parent.parent
sys.path.append(str(root))
from ClassicalMSS.utils.svd import power_iteration, power_iteration_old

def test_power_iteration_methods():
    """Compare power iteration implementations and torch.svd_lowrank for accuracy, speed, and stability."""
    
    # Simplified test configurations
    matrix_sizes = [1000]
    num_trials = 10  # For stability testing
    batch_sizes = [3] # The number of parallel starting points
    num_iterations = [100]  # The number of iterations
    
    # Create data structures to store results for plotting
    results_data = {
        'Matrix Size': [], 'Batch Size': [], 'Iterations': [],
        'Method': [], 'Error': [], 'Time (ms)': [], 'Std Dev': []
    }
    
    for size in matrix_sizes:
        print(f"\nTesting matrix size: {size}x{size}")
        
        # Create a test matrix with known maximum eigenvalue
        # Using a symmetric matrix for simplicity (eigenvalues are real)
        matrix = torch.randn(size, size)
        matrix = matrix @ matrix.T  # Make it symmetric positive definite
        
        # Get true maximum eigenvalue using torch.linalg.eigvalsh
        # Use eigvalsh instead of eigvals since we know the matrix is symmetric
        # Also, we only need the largest eigenvalue, so we can use 'largest' option
        true_max_eigenval = torch.linalg.eigvalsh(matrix)[-1].item()
        
        for bs in batch_sizes:
            for iters in num_iterations:
                print(f"\nBatch size: {bs}, Iterations: {iters}")
                
                # Test stability and timing for both methods
                old_results = []
                new_results = []
                old_times = []
                new_times = []
                
                for trial in range(num_trials):
                    # Test old method
                    start_time = time.time()
                    old_result = power_iteration_old(matrix, num_iters=iters, bs=bs)
                    old_times.append(time.time() - start_time)
                    old_results.append(old_result.item())
                    
                    # Test new method
                    start_time = time.time()
                    new_result = power_iteration(matrix, num_iters=iters, bs=bs)
                    new_times.append(time.time() - start_time)
                    new_results.append(new_result.item())
                
                # Calculate statistics
                old_mean = np.mean(old_results)
                old_std = np.std(old_results)
                new_mean = np.mean(new_results)
                new_std = np.std(new_results)
                
                print(f"True maximum eigenvalue: {true_max_eigenval:.4f}")
                print("Old method:")
                print(f"  Mean: {old_mean:.4f} (error: {abs(old_mean-true_max_eigenval):.4f})")
                print(f"  Std:  {old_std:.4f}")
                print(f"  Time: {np.mean(old_times)*1000:.2f}ms ± {np.std(old_times)*1000:.2f}ms")
                print("New method:")
                print(f"  Mean: {new_mean:.4f} (error: {abs(new_mean-true_max_eigenval):.4f})")
                print(f"  Std:  {new_std:.4f}")
                print(f"  Time: {np.mean(new_times)*1000:.2f}ms ± {np.std(new_times)*1000:.2f}ms")
                
                # Test torch.svd_lowrank
                torch_results = []
                torch_times = []
                for trial in range(num_trials):
                    start_time = time.time()
                    # Get largest singular value using torch.svd_lowrank
                    U, S, Vh = torch.svd_lowrank(matrix, q=1, niter=iters)
                    torch_result = S[0].item()
                    torch_times.append(time.time() - start_time)
                    torch_results.append(torch_result)
                
                torch_mean = np.mean(torch_results)
                torch_std = np.std(torch_results)
                
                # Store results for plotting
                for method, mean_val, std_val, times in [
                    ('Old Power Iteration', old_mean, old_std, old_times),
                    ('New Power Iteration', new_mean, new_std, new_times),
                    ('torch.svd_lowrank', torch_mean, torch_std, torch_times)
                ]:
                    results_data['Matrix Size'].append(size)
                    results_data['Batch Size'].append(bs)
                    results_data['Iterations'].append(iters)
                    results_data['Method'].append(method)
                    results_data['Error'].append(abs(mean_val - true_max_eigenval))
                    results_data['Time (ms)'].append(np.mean(times) * 1000)
                    results_data['Std Dev'].append(std_val)
                
                # Print torch.svd_lowrank results
                print("torch.svd_lowrank:")
                print(f"  Mean: {torch_mean:.4f} (error: {abs(torch_mean-true_max_eigenval):.4f})")
                print(f"  Std:  {torch_std:.4f}")
                print(f"  Time: {np.mean(torch_times)*1000:.2f}ms ± {np.std(torch_times)*1000:.2f}ms")

    # Create visualizations
    plot_results(results_data)

def plot_results(data):
    """Simplified visualizations comparing the three methods."""
    import pandas as pd
    df = pd.DataFrame(data)
    
    plt.style.use('seaborn-v0_8')
    
    # Create a single figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Time Comparison (Box Plot)
    sns.boxplot(data=df, x='Method', y='Time (ms)', ax=axes[0])
    axes[0].set_title('Computation Time Comparison')
    axes[0].set_yscale('log')
    axes[0].tick_params(axis='x', rotation=45)
    
    # 2. Error Comparison (Box Plot)
    sns.boxplot(data=df, x='Method', y='Error', ax=axes[1])
    axes[1].set_title('Error Comparison')
    axes[1].set_yscale('log')
    axes[1].tick_params(axis='x', rotation=45)
    
    # 3. Stability Comparison (Box Plot)
    sns.boxplot(data=df, x='Method', y='Std Dev', ax=axes[2])
    axes[2].set_title('Stability Comparison')
    axes[2].set_yscale('log')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.suptitle(f'Method Comparison (Matrix Size: 1000x1000, Batch Size: 10, Iterations: 8)', y=1.05)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_power_iteration_methods()

