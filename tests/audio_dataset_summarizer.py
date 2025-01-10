"""
Summarize an audio dataset on its audio properties distribution.
The dataset should be arranged in numbered folders,
each containing a corresponding number_mixed.wav which will be the target audio for inspection.

The datasets in the training folders are already summarized and the summary can be found in the same parent folder
"""

import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

root_dir = Path(__file__).parents[1]
sys.path.append(str(root_dir))  # Add project root to path
from ClassicalMSS.datasets_processing.Utils_audio_inspector import inspect_audio


dataset_sample_rate = 48000
#dataset_dir = root_dir / "datasets" / "1_midi_generated" / "audio_dataset" / f"{dataset_sample_rate}Hz"
dataset_dir = root_dir / "datasets" / "3_youtube_test" / "audio_dataset" / f"{dataset_sample_rate}Hz"

def collect_audio_stats():
    stats = {
        'sample_rates': [],
        'bit_depths': [],
        'durations': [],
        'num_frames': [],
        'rms_dBFS': [],
        'peak_dBFS': [],
        'LUFS': []
    }
    
    # Iterate through all numbered folders
    for folder in os.listdir(dataset_dir):
        if folder.isdigit():
            mix_file = os.path.join(dataset_dir, folder, f"{folder}_mixed.wav")
            if os.path.exists(mix_file):
                # Get audio stats using inspect_audio
                sr, bd, dur, nf, rms, peak, lufs = inspect_audio(mix_file, print_to_console=False)
                
                stats['sample_rates'].append(sr)
                stats['bit_depths'].append(bd)
                stats['durations'].append(dur)
                stats['num_frames'].append(nf)
                stats['rms_dBFS'].append(rms)
                stats['peak_dBFS'].append(peak)
                stats['LUFS'].append(lufs)
                
    return stats

def plot_distributions(stats):
    # Create two separate figures: one for expected fixed values, one for variable measures
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
    fig1.suptitle('Parameters with Expected Fixed Values')
    
    # Plot fixed-value parameters using box plots
    axes1[0].boxplot(stats['durations'])
    axes1[0].set_title('Duration Distribution\n(Expected: 5s)')
    axes1[0].set_ylabel('Seconds')
    # Highlight outliers
    outliers = [x for x in stats['durations'] if abs(x - 5.0) > 0.1]
    if outliers:
        print(f"\nWARNING: Found {len(outliers)} duration outliers: {outliers}")
    
    axes1[1].boxplot(stats['bit_depths'])
    axes1[1].set_title('Bit Depth Distribution\n(Expected: 16)')
    # Highlight outliers
    outliers = [x for x in stats['bit_depths'] if x != 16]
    if outliers:
        print(f"\nWARNING: Found {len(outliers)} bit depth outliers: {outliers}")
    
    axes1[2].boxplot(stats['sample_rates'])
    axes1[2].set_title('Sample Rate Distribution\n(Expected: 16000)')
    # Highlight outliers
    outliers = [x for x in stats['sample_rates'] if x != dataset_sample_rate]
    if outliers:
        print(f"\nWARNING: Found {len(outliers)} sample rate outliers: {outliers}")
    
    plt.tight_layout()
    
    # Create second figure for variable measures
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    fig2.suptitle('Audio Level Distributions')
    
    # Plot level measurements using histograms
    axes2[0].hist(stats['rms_dBFS'], bins=20)
    axes2[0].set_title('RMS dBFS Distribution')
    
    axes2[1].hist(stats['peak_dBFS'], bins=20)
    axes2[1].set_title('Peak dBFS Distribution')
    
    axes2[2].hist(stats['LUFS'], bins=20)
    axes2[2].set_title('LUFS Distribution')
    
    plt.tight_layout()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for key, values in stats.items():
        values = np.array(values)
        print(f"\n{key}:")
        print(f"Mean Value: {np.mean(values):.2f}")
        print(f"Standard Deviation: {np.std(values):.2f}")
        print(f"Minimum Value: {np.min(values):.2f}")
        print(f"Maximum Value: {np.max(values):.2f}")

def main():
    stats = collect_audio_stats()
    plot_distributions(stats)
    plt.show()

if __name__ == "__main__":
    main()


