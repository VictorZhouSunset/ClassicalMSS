import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
import time
from tqdm import tqdm
from google.colab import drive
import sys

# Check if running in Colab
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    drive.mount('/content/drive')
    # Add the path to your project folder in Google Drive
    project_path = "/content/drive/MyDrive/2024-2025 GapYear/Demucs_Sound Source Separation/Try_Simulate_Violin_Piano"
    sys.path.append(project_path)

from Models.WUN.Wave_U_Net_Mono import WaveUNet_Mono  # Import your model class from the original file


# Custom dataset
class AudioDataset(Dataset):
    def __init__(self, data_dir, start_idx, end_idx, step=1, fixed_length=48000*12):
        self.data_dir = data_dir
        self.file_indices = list(range(start_idx, end_idx + 1, step))
        self.fixed_length = fixed_length

    def __len__(self):
        return len(self.file_indices)

    def __getitem__(self, idx):
        file_idx = self.file_indices[idx]
        mixed_path = os.path.join(self.data_dir, str(file_idx), f"{file_idx}_mixed.wav")
        violin_path = os.path.join(self.data_dir, str(file_idx), f"{file_idx}_violin.wav")
        piano_path = os.path.join(self.data_dir, str(file_idx), f"{file_idx}_piano.wav")

        mixed, _ = torchaudio.load(mixed_path)
        violin, _ = torchaudio.load(violin_path)
        piano, _ = torchaudio.load(piano_path)

        # Convert to mono if stereo
        mixed = self.to_mono(mixed)
        violin = self.to_mono(violin)
        piano = self.to_mono(piano)

        # Pad or truncate to fixed length
        mixed = self.pad_or_truncate(mixed)
        violin = self.pad_or_truncate(violin)
        piano = self.pad_or_truncate(piano)

        return mixed, violin, piano

    def to_mono(self, audio):
        if audio.shape[0] > 1:
            return audio.mean(dim=0, keepdim=True)
        return audio

    def pad_or_truncate(self, tensor):
        if tensor.size(1) > self.fixed_length:
            return tensor[:, :self.fixed_length]
        elif tensor.size(1) < self.fixed_length:
            padding = self.fixed_length - tensor.size(1)
            return nn.functional.pad(tensor, (0, padding))
        else:
            return tensor

# Training function
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for mixed, violin, piano in progress_bar:
        mixed, violin, piano = mixed.to(device), violin.to(device), piano.to(device)
        
        optimizer.zero_grad()
        output = model(mixed)
        loss = criterion(output[:, 0, :], violin.squeeze(1)) + criterion(output[:, 1, :], piano.squeeze(1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
    progress_bar.close()
    return total_loss / len(train_loader)

# Evaluation function
def evaluate(model, eval_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        progress_bar = tqdm(eval_loader, desc="Evaluating", leave=False)
        for mixed, violin, piano in progress_bar:
            mixed, violin, piano = mixed.to(device), violin.to(device), piano.to(device)
            
            output = model(mixed)
            loss = criterion(output[:, 0, :], violin.squeeze(1)) + criterion(output[:, 1, :], piano.squeeze(1))
            
            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        progress_bar.close()
    return total_loss / len(eval_loader)

def get_num_workers():
    if IN_COLAB:
        return 2 if torch.cuda.is_available() else 1
    else:
        return min(6, os.cpu_count())  # Use up to 6 workers locally, or less if fewer cores are available
    
# Main training loop
def run_model(model):
    print("Starting model creation")
    data_dir = "Data/training_data/normal_data"
    eval_dir = "Data/evaluation_data"
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
        train_dataset = AudioDataset(data_dir, 3, 1599, step=4, fixed_length=48000*12)  # Adjust fixed_length as needed
        in_group_eval_dataset = AudioDataset(data_dir, 1603, 1703, step=4, fixed_length=48000*12)
        out_group_eval_dataset = AudioDataset(eval_dir, 0, 29, step=1, fixed_length=48000*12)
        
        num_workers = get_num_workers()
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=num_workers)
        in_group_eval_loader = DataLoader(in_group_eval_dataset, batch_size=4, shuffle=False, num_workers=num_workers)
        out_group_eval_loader = DataLoader(out_group_eval_dataset, batch_size=4, shuffle=False, num_workers=num_workers)
        # Training loop
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        num_epochs = 20
        for epoch in range(num_epochs):
            start_time = time.time()
            train_loss = train(model, train_loader, optimizer, criterion, device)
            in_group_eval_loss = evaluate(model, in_group_eval_loader, criterion, device)
            out_group_eval_loss = evaluate(model, out_group_eval_loader, criterion, device)
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, In-Group Eval Loss: {in_group_eval_loss:.4f}, Out-Group Eval Loss: {out_group_eval_loss:.4f}, Time: {epoch_time:.2f}s")
        
        if IN_COLAB:
            save_path = os.path.join(project_path, f"{model.name}_training.pth")
        else:
            save_path = f"{model.name}_training.pth"
        
        # Move model to CPU before saving
        model_state_dict = model.cpu().state_dict()
        torch.save(model_state_dict, save_path)
        print(f"Model saved to {save_path}")
        
        # Move model back to the original device if you need to continue using it
        model = model.to(device)
        
    except RuntimeError as e:
        print(f"Error during model creation: {e}")

model = WaveUNet_Mono(num_layers=4, num_initial_filters=8)
print("Model created successfully")
run_model(model)
