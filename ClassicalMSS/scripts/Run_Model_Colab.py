import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from google.colab import drive
import sys
from torch.optim.lr_scheduler import LambdaLR
import math

# Check if running in Colab
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    drive.mount('/content/drive')
    # Add the path to your project folder in Google Drive
    project_path = "/content/drive/MyDrive/2024-2025 GapYear/Demucs_Sound Source Separation/Try_Simulate_Violin_Piano"
    sys.path.append(project_path)

from Models.WUN.Wave_U_Net import WaveUNet
from Models.WUN.Wave_U_Net_Dropout import WaveUNet_Dropout
from Models.Conv_Tasnet.Conv_Tasnet import Conv_Tasnet
from Models.Conv_Tasnet.Conv_Tasnet_tanh import Conv_Tasnet_Tanh
from Models.Demucs.Demucs_Basic import Demucs_Basic
from Models.Demucs.Demucs_LSTM_Inverted import Demucs_LSTM_Inverted
from Models.Demucs.Demucs_Smooth_Decay import Demucs_Smooth_Decay
from Models.U_Light.U_Light_Raw import U_Light_Raw

# Custom dataset
class AudioDataset(Dataset):
    def __init__(self, data_dir, start_idx, end_idx, step=1, fixed_length=48000*12):
        self.data_dir = data_dir
        self.file_indices = list(range(start_idx, end_idx + 1, step))
        self.fixed_length = fixed_length
        self.sample_rates = [16000,22050,44100,48000]

    def __len__(self):
        return len(self.file_indices)

    def __getitem__(self, idx):
        file_idx = self.file_indices[idx]
        # Check for different possible file names with various sample rates
        possible_mixed_names = [f"{file_idx}_mixed.wav"] + [f"{file_idx}_{rate}Hz_mixed.wav" for rate in self.sample_rates] + [f"{file_idx}_mixed_{rate}Hz.wav" for rate in self.sample_rates]
        possible_violin_names = [f"{file_idx}_violin.wav"] + [f"{file_idx}_{rate}Hz_violin.wav" for rate in self.sample_rates] + [f"{file_idx}_violin_{rate}Hz.wav" for rate in self.sample_rates]
        possible_piano_names = [f"{file_idx}_piano.wav"] + [f"{file_idx}_{rate}Hz_piano.wav" for rate in self.sample_rates] + [f"{file_idx}_piano_{rate}Hz.wav" for rate in self.sample_rates]

        mixed_path = None
        violin_path = None
        piano_path = None

        for name in possible_mixed_names:
            path = os.path.join(self.data_dir, str(file_idx), name)
            if os.path.exists(path):
                mixed_path = path
                break

        for name in possible_violin_names:
            path = os.path.join(self.data_dir, str(file_idx), name)
            if os.path.exists(path):
                violin_path = path
                break

        for name in possible_piano_names:
            path = os.path.join(self.data_dir, str(file_idx), name)
            if os.path.exists(path):
                piano_path = path
                break

        if mixed_path is None or violin_path is None or piano_path is None:
            raise FileNotFoundError(f"Could not find all required audio files for index {file_idx}, mixed: {mixed_path}, violin: {violin_path}, piano: {piano_path}")

        mixed, _ = torchaudio.load(mixed_path)
        violin, _ = torchaudio.load(violin_path)
        piano, _ = torchaudio.load(piano_path)

        # Pad or truncate to fixed length
        mixed = self.pad_or_truncate(mixed)
        violin = self.pad_or_truncate(violin)
        piano = self.pad_or_truncate(piano)

        return mixed, violin, piano

    def pad_or_truncate(self, tensor):
        if tensor.size(1) > self.fixed_length:
            return tensor[:, :self.fixed_length]
        elif tensor.size(1) < self.fixed_length:
            padding = self.fixed_length - tensor.size(1)
            return nn.functional.pad(tensor, (0, padding))
        else:
            return tensor

# Function to plot gradient flow
def plot_grad_flow(named_parameters, epoch, batch=None, n_layers_to_show=10):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().item())
    plt.figure(figsize=(10, 8))
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    
    # Show only every nth label
    n = max(1, len(layers) // n_layers_to_show)
    plt.xticks(range(0, len(ave_grads), n), [layers[i] for i in range(0, len(layers), n)], rotation="vertical")
    
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.title(f"Gradient flow - Epoch {epoch+1}" + (f" Batch {batch}" if batch else ""))
    plt.grid(True)
    
    # Create a directory for gradient flow plots if it doesn't exist
    os.makedirs(os.path.join(project_path, 'gradient_flow_plots'), exist_ok=True)
    
    # Save the plot with a unique name
    if batch is not None:
        filename = os.path.join(project_path, f'gradient_flow_plots/gradient_flow_epoch_{epoch+1}_batch_{batch}.png')
    else:
        filename = os.path.join(project_path, f'gradient_flow_plots/gradient_flow_epoch_{epoch+1}_final.png')
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gradient flow plot saved: {filename}")

# Modify the train function
def train(model, train_loader, optimizer, criterion, cross_loss, device, epoch, verbose=False):
    model.train()
    total_loss = 0
    total_samples = 0
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
    for i, (mixed, violin, piano) in enumerate(progress_bar):
        batch_size = mixed.size(0)
        mixed, violin, piano = mixed.to(device), violin.to(device), piano.to(device)
        
        optimizer.zero_grad()
        output = model(mixed)
        loss = criterion(output[:, 0:2, :], violin) + criterion(output[:, 2:4, :], piano)
        if cross_loss:
            loss -= criterion(output[:, 0:2, :], piano) + criterion(output[:, 2:4, :], violin)
        loss.backward()
        
        # Plot gradient flow every 100 batches if verbose
        if verbose and (i + 1) % 100 == 0:
            progress_bar.clear()
            plot_grad_flow(model.named_parameters(), epoch, i + 1)
            tqdm.write(f"Gradient flow plot saved: gradient_flow_plots/gradient_flow_epoch_{epoch+1}_batch_{i+1}.png")
        
        optimizer.step()
        
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    progress_bar.close()
    # print(f"Train Total Loss: {total_loss :.4f}; Total Samples: {total_samples}")
    return total_loss / total_samples

# Evaluation function
def evaluate(model, eval_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    progress_bar = tqdm(eval_loader, desc="Evaluating", leave=False)
    print(f"\nStarting Evaluation, total loss: {total_loss:.4f}, total samples: {total_samples}")
    with torch.no_grad():
        for mixed, violin, piano in progress_bar:
            batch_size = mixed.size(0)
            mixed, violin, piano = mixed.to(device), violin.to(device), piano.to(device)
            
            output = model(mixed)
            violin_result = output[:, 0:2, :]
            piano_result = output[:, 2:4, :]
            loss_violin = criterion(violin_result, violin)
            loss_piano = criterion(piano_result, piano)
            # if loss_violin.item() < -10 or loss_piano.item() < -10:
            #     # Save the violin_result and piano_result as wave files
            #     for i in range(batch_size):
            #         torchaudio.save(f'violin_result_batch_{total_samples}_sample_{i}.wav', violin_result[i].cpu(), 16000)
            #         torchaudio.save(f'piano_result_batch_{total_samples}_sample_{i}.wav', piano_result[i].cpu(), 16000)
            #     print(f"Saved abnormal results as wave files for batch {total_samples}")
                
            # print(f"\nViolin Loss: {loss_violin.item():.4f}, Piano Loss: {loss_piano.item():.4f}")

            loss = loss_violin + loss_piano
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # print(f"\nEval Loss: {loss.item():.4f}, Total Loss: {total_loss:.4f}")
            # print(f"Current Batch Size: {batch_size}, Total Samples: {total_samples}")

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    progress_bar.close()
    print(f"Eval Total Loss: {total_loss :.4f}; Total Samples: {total_samples}")
    return total_loss / total_samples

def si_snr_loss(estimated, target):
    def l2norm(tensor):
        return torch.sqrt(torch.sum(tensor**2, dim=2, keepdim=True) + 1e-8)
    
    # Ensure input tensors have shape (batch_size, 2, samples)
    assert estimated.dim() == 3 and target.dim() == 3, "Input tensors must be 3D (batch_size, channels, samples)"
    assert estimated.size(1) == 2 and target.size(1) == 2, "Input tensors must have 2 channels (for violin or piano)"
    
    # Step 1. Zero-mean norm
    mean_estimated = torch.mean(estimated, dim=2, keepdim=True)
    mean_target = torch.mean(target, dim=2, keepdim=True)
    estimated = estimated - mean_estimated
    target = target - mean_target

    # Step 2. SI-SNR
    dot_product = torch.sum(estimated * target, dim=2, keepdim=True)
    s_target = dot_product * target / (l2norm(target)**2 + 1e-8)
    e_noise = estimated - s_target

    si_snr = 20 * torch.log10(l2norm(s_target) / (l2norm(e_noise) + 1e-8))
    return -torch.mean(si_snr)

def get_num_workers():
    if IN_COLAB:
        return 2 if torch.cuda.is_available() else 1
    else:
        return min(6, os.cpu_count())  # Use up to 6 workers locally, or less if fewer cores are available

# Add this function to create the learning rate scheduler
def get_lr_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    
    return LambdaLR(optimizer, lr_lambda)

# Main training loop
def run_model(model, num_epochs, warmup_epochs, loss_function, lr=0.001, small_size=False, verbose=False):
    setup_start_time = time.time()  # Start the setup timer
    print("Starting model creation")

    if not small_size:
      if IN_COLAB:
        data_dir = os.path.join(project_path, "Data", "training_data", "normal_data")
        eval_dir = os.path.join(project_path, "Data", "evaluation_data")
      else:
        data_dir = "Data/training_data/normal_data"
        eval_dir = "Data/evaluation_data"
    else:
      if IN_COLAB:
        data_dir = os.path.join(project_path, "Data", "training_data", "small_size")
        eval_dir = os.path.join(project_path, "Data", "evaluation_data_small_size")
      else:
        data_dir = "Data/training_data/small_size"
        eval_dir = "Data/evaluation_data_small_size"




    print("Data directory: ", data_dir)
    print("Evaluation directory: ", eval_dir)
    print(f"CUDA is {'available' if torch.cuda.is_available() else 'not available'}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")

    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    try:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params}")

        # Create dataset and dataloader
        if small_size:
            data_sample_rate = 16000
            data_length_in_seconds = 5
        else:
            data_sample_rate = 48000
            data_length_in_seconds = 12
        fixed_data_length = data_sample_rate * data_length_in_seconds

        if small_size:
            train_dataset = AudioDataset(data_dir, 0, 999, step=1, fixed_length=fixed_data_length)
            in_group_eval_dataset = AudioDataset(data_dir, 4000, 4029, step=1, fixed_length=fixed_data_length)
            out_group_eval_dataset = AudioDataset(eval_dir,  0, 29, step=1, fixed_length=fixed_data_length)
        else:
            train_dataset = AudioDataset(data_dir, 3, 1599, step=4, fixed_length=fixed_data_length)
            in_group_eval_dataset = AudioDataset(data_dir, 1603, 1703, step=4, fixed_length=fixed_data_length)
            out_group_eval_dataset = AudioDataset(eval_dir, 0, 29, step=1, fixed_length=fixed_data_length)


        # Use this function when creating your DataLoaders
        num_workers = get_num_workers()
        if small_size:
            batch_size = 100;
        else:
            batch_size = 2;
        eval_batch_size = 10;
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        in_group_eval_loader = DataLoader(in_group_eval_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)
        out_group_eval_loader = DataLoader(out_group_eval_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers)

        # Training loop
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Add the learning rate scheduler
        if warmup_epochs != -1:
            scheduler = get_lr_scheduler(optimizer, warmup_epochs, num_epochs)

        if loss_function == "mse" or loss_function == "l2":
            criterion = nn.MSELoss()
        elif loss_function == "si_snr" or loss_function == "si_snr_2":
            criterion = si_snr_loss
        elif loss_function == "l1":
            criterion = nn.L1Loss()
        else:
            raise ValueError(f"Invalid loss function: {loss_function}")

        eval_criterion = si_snr_loss

        setup_end_time = time.time()  # End the setup timer
        setup_time = setup_end_time - setup_start_time
        print(f"Setup time (before training): {setup_time:.2f} seconds")

        num_epochs = num_epochs if small_size else 6
        best_out_group_sisnr = float('inf')
        best_in_group_sisnr = float('inf')
        best_out_group_model_state = None
        best_in_group_model_state = None

        for epoch in range(num_epochs):
            start_time = time.time()
            if loss_function == "si_snr_2":
                cross_loss = True
            else:
                cross_loss = False
            train_loss = train(model, train_loader, optimizer, criterion, cross_loss, device, epoch, verbose)
            
            if warmup_epochs != -1:
                # Step the scheduler
                scheduler.step()
                
                # Log the current learning rate
                current_lr = scheduler.get_last_lr()[0]
                print(f"Current learning rate: {current_lr:.6f}")
            
            if verbose:
                log_weight_stats(model, epoch)

            in_group_eval_loss = evaluate(model, in_group_eval_loader, criterion, device)
            in_group_eval_SiSNR = evaluate(model, in_group_eval_loader, eval_criterion, device)
            out_group_eval_loss = evaluate(model, out_group_eval_loader, criterion, device)
            out_group_eval_SiSNR = evaluate(model, out_group_eval_loader, eval_criterion, device)
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, In-Group Eval Loss: {in_group_eval_loss:.4f}, Out-Group Eval Loss: {out_group_eval_loss:.4f}, Time: {epoch_time:.2f}s")
            print(f"In-Group Eval SiSNR: {in_group_eval_SiSNR:.4f}, Out-Group Eval SiSNR: {out_group_eval_SiSNR:.4f}")

            # Check if this is the best model so far
            if out_group_eval_SiSNR < best_out_group_sisnr:
                best_out_group_sisnr = out_group_eval_SiSNR
                best_out_group_model_state = model.cpu().state_dict()
                model = model.to(device)  # Move model back to device
                print(f"New best model found at epoch {epoch+1} with Out-Group Eval SiSNR: {best_out_group_sisnr:.4f}")
            
            if in_group_eval_SiSNR < best_in_group_sisnr:
                best_in_group_sisnr = in_group_eval_SiSNR
                best_in_group_model_state = model.cpu().state_dict()
                model = model.to(device)  # Move model back to device
                print(f"New best model found at epoch {epoch+1} with In-Group Eval SiSNR: {best_in_group_sisnr:.4f}")
            
            if small_size:
                tmp_save_path = os.path.join(project_path, f"{model.name}_tmp_small_size.pth") if IN_COLAB else f"{model.name}_tmp_small_size.pth"
                torch.save(model.state_dict(), tmp_save_path)
            else:
                tmp_save_path = os.path.join(project_path, f"{model.name}_tmp.pth") if IN_COLAB else f"{model.name}_tmp.pth"
                torch.save(model.state_dict(), tmp_save_path)

        # Save the best model
        if small_size:
            in_group_save_path = os.path.join(project_path, f"{model.name}_small_size.pth") if IN_COLAB else f"{model.name}_small_size.pth"
            out_group_save_path = os.path.join(project_path, f"{model.name}_small_size_out_group.pth") if IN_COLAB else f"{model.name}_small_size_out_group.pth"
        else:
            in_group_save_path = os.path.join(project_path, f"{model.name}.pth") if IN_COLAB else f"{model.name}.pth"
            out_group_save_path = os.path.join(project_path, f"{model.name}_out_group.pth") if IN_COLAB else f"{model.name}_out_group.pth"

        torch.save(best_in_group_model_state, in_group_save_path)
        torch.save(best_out_group_model_state, out_group_save_path)
        print(f"Best models saved to {in_group_save_path} and {out_group_save_path}")

        # Move model back to the original device if you need to continue using it
        model = model.to(device)

    except RuntimeError as e:
        print(f"Error during model creation: {e}")

def log_weight_stats(model, epoch):
    print(f"Epoch {epoch} Weight Statistics:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            max_val = param.data.max().item()
            min_val = param.data.min().item()
            mean_val = param.data.mean().item()
            std_val = param.data.std().item()
            
            # Only print if there are potentially problematic values
            if abs(max_val) > 10 or abs(min_val) > 10 or abs(mean_val) > 1 or std_val > 1:
                print(f"  Layer: {name}")
                print(f"    Mean: {mean_val:.4f}, Std: {std_val:.4f}")
                print(f"    Max: {max_val:.4f}, Min: {min_val:.4f}")

# model = WaveUNet(num_layers=4, num_initial_filters=8, dropout_rate=0.1)
# model = WaveUNet_Dropout(num_layers=4, num_initial_filters=8, dropout_rate=0.9)
# model = Demucs_Basic(L=6, K=8, S=4, C0=2, C1=64, D=1, LSTM_layers=2, all_conv=False)
# model = Demucs_LSTM_Inverted(L=3, K=8, S=4, C0=2, C1=4, D=1, LSTM_layers=2, all_conv=False)
# model = Demucs_Smooth_Decay(L=6, K=8, S=4, C0=2, C1=64, D=4, LSTM_layers=2, all_conv=False)
model = U_Light_Raw(num_layers=3, num_filters=8, kernel_size=15, tanh=True, dropout_rate=0.1)
# model = Conv_Tasnet(N=8, small_size=True)
print("Model created successfully")
run_model(model, num_epochs=5, warmup_epochs=1, loss_function="mse", small_size=True, verbose=False)
