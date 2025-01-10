"""
This script is used to inspect audio files and print the audio parameters.
Supports multiple formats including WAV and MP3.
"""
import numpy as np
import pyloudnorm as pyln
from tkinter import filedialog
import tkinter as tk
import soundfile as sf
import os

def inspect_audio(audio_file_path=None, print_to_console=True):
    # If no path is provided, open file dialog
    if audio_file_path is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        audio_file_path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.ogg"),
                ("WAV files", "*.wav"),
                ("MP3 files", "*.mp3"),
                ("All files", "*.*")
            ]
        )
        
        if not audio_file_path:  # If user cancels selection
            print("No file selected")
            return

    # Read audio file using soundfile
    audio_data, sample_rate = sf.read(audio_file_path)
    
    # Get basic parameters
    num_frames = len(audio_data)
    duration = num_frames / sample_rate
    num_channels = 1 if len(audio_data.shape) == 1 else audio_data.shape[1]
    
    # Convert to float32 if not already
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    # Calculate audio metrics
    rms_dBFS = 20 * np.log10(np.sqrt(np.mean(audio_data**2)) + 1e-8)
    peak_dBFS = 20 * np.log10(np.max(np.abs(audio_data) + 1e-8))
    
    # Ensure correct shape for LUFS calculation
    if len(audio_data.shape) == 1:
        audio_data = audio_data.reshape(-1, 1)
    
    # Create meter and measure loudness
    meter = pyln.Meter(sample_rate)
    LUFS = meter.integrated_loudness(audio_data)
    
    # Get bit depth from file info (if available)
    try:
        info = sf.info(audio_file_path)
        # Map common soundfile subtypes to bit depths
        subtype_to_bits = {
            'PCM_16': 16,
            'PCM_24': 24,
            'PCM_32': 32,
            'FLOAT': 32,
            'DOUBLE': 64,
        }
        bit_depth = subtype_to_bits.get(info.subtype, 16)  # default to 16 if unknown
    except:
        bit_depth = -1  # indicating unknown
    
    if print_to_console:
        print(f"File: {os.path.basename(audio_file_path)}")
        print(f"Number of channels: {num_channels}")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Bit depth: {bit_depth} bits")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Number of frames: {num_frames}")
        print(f"RMS dBFS: {rms_dBFS:.1f} dB")
        print(f"Peak dBFS: {peak_dBFS:.1f} dB")
        print(f"LUFS: {LUFS:.1f} dB")
        
    return sample_rate, bit_depth, duration, num_frames, rms_dBFS, peak_dBFS, LUFS

def main():
    inspect_audio()  # This will open the file dialog since no path is provided

if __name__ == "__main__":
    main()
