"""
This file contains utility functions for audio preprocessing, including:
- Combining two audio arrays by aligning at beginning and padding shorter one.
- Trimming or padding audio to target duration following specified logic.
- Normalizing and compressing audio dynamics.
- Converting stereo audio to mono by averaging channels.
"""

import numpy as np
import wave
import librosa
from scipy.signal import resample as scipy_resample
import resampy
import soxr
import shutil

def combine_audios(audio1, audio2, sr):
    """Combine two audio arrays by aligning at beginning and padding shorter one.
    
    Args:
        audio1: First audio array
        audio2: Second audio array 
        sr: Sample rate
    Returns:
        Combined audio array
    """
    if len(audio1) > len(audio2):
        # Pad audio2 with zeros
        padded = np.pad(audio2, (0, len(audio1) - len(audio2)))
        return (audio1 + padded) / 2
    elif len(audio2) > len(audio1):
        # Pad audio1 with zeros
        padded = np.pad(audio1, (0, len(audio2) - len(audio1)))
        return (padded + audio2) / 2
    else:
        return (audio1 + audio2) / 2

def trim_or_pad(audio, sr, target_duration=5.0, silence_threshold=0.001):
    """Trim or pad audio to target duration following specified logic.
    
    Args:
        audio: Input audio array (each sample is float in range -1 to 1)
        sr: Sample rate
        target_duration: Target duration in seconds (default 5.0)
        silence_threshold: Threshold for silence detection
    Returns:
        Processed audio array
    """
    target_length = int(sr * target_duration)
    current_length = len(audio)
    
    if current_length < target_length:
        # Pad if shorter
        pad_length = target_length - current_length
        pad_start = pad_length // 2
        pad_end = pad_length - pad_start
        return np.pad(audio, (pad_start, pad_end))
        
    elif current_length > target_length:
        # Trim if longer
        # Find silence at start and end
        start_silence = 0
        end_silence = 0
        
        # Check silence from start
        while start_silence < len(audio) and np.abs(audio[start_silence]) < silence_threshold:
            start_silence += 1
            
        # Check silence from end
        while end_silence < len(audio) and np.abs(audio[-(end_silence+1)]) < silence_threshold:
            end_silence += 1
        
        # First trim silence from both ends, if shorter than target duration, pad with silence
        if start_silence > 0 or end_silence > 0:
            trim_start = start_silence
            trim_end = end_silence
            audio = audio[trim_start:-trim_end]
            audio = trim_or_pad(audio, sr, target_duration, silence_threshold)
        else: # If still too long, trim from end
            audio = audio[:target_length]
                        
    return audio

def normalize_dynamics(audio, target_db=-15, threshold_db=-24):
    """Normalize and then compress audio dynamics.
    
    Args:
        audio: Input audio array (float32 in range -1 to 1)
        target_db: Target RMS level in dB
        threshold_db: Compression threshold in dB
    Returns:
        Processed audio array (float32)
    """
    # First normalize to target RMS
    current_rms = np.sqrt(np.mean(audio**2))
    target_rms = 10 ** (target_db/20)
    audio = audio * (target_rms / current_rms)
    
    # Convert to dB for compression
    db = 20 * np.log10(np.abs(audio) + 1e-8)
    
    # Apply compression to avoid clipping
    if np.max(db) > 0:
        ratio = (np.max(db) - threshold_db) / (0 - threshold_db)
        mask = db > threshold_db
        db[mask] = threshold_db + (db[mask] - threshold_db) / ratio
    
    # Convert back to linear
    audio = np.sign(audio) * (10 ** (db/20))
    
    # Final clip check
    max_val = np.max(np.abs(audio))
    if max_val > 1:
        audio = audio / max_val
        
    return audio.astype(np.float32)

def stereo_to_mono(audio):
    """Convert stereo audio to mono by averaging channels.
    
    Args:
        audio: Input stereo audio array (n_samples, 2)
    Returns:
        Mono audio array (n_samples,)
    """
    if len(audio.shape) == 2 and audio.shape[1] == 2:
        return np.mean(audio, axis=1)
    return audio

def copy_file(src, dst):
    """Copy a file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination file path
    """
    shutil.copy(src, dst)

def get_silence_proportion(audio, threshold_db=-60):
    """Get the proportion of silence in the audio.
    
    Args:
        audio: Input audio array (float32 in range -1 to 1)
        threshold_db: Threshold for silence detection
    Returns:
        Proportion of silence in the audio
    """
    db = 20 * np.log10(np.abs(audio) + 1e-8)
    return np.mean(db < threshold_db)

def segment_audio(audio, sr, segment_length_seconds=5.0, segment_window_interval_seconds=2.5, output_mono=False):
    """Segment audio into clips of specified length and overlap.
    
    Args:
        audio: Input audio array (float32 in range -1 to 1)
        segment_length_seconds: Length of each segment in seconds
        segment_window_interval_seconds: The distance between the start of each segment and the start of the next segment
        output_mono: Whether to convert to mono
    Returns:
        List of processed audio clips
    """
    # Convert to mono if requested
    if output_mono:
        audio = stereo_to_mono(audio)

    # Get audio parameters
    total_sample_count = audio.shape[0]

    # Calculate segment parameters in samples
    segment_length = int(segment_length_seconds * sr)
    segment_interval = int(segment_window_interval_seconds * sr)

    # Initialize list for processed clips
    segments = []

    # Segment the audio
    start = 0
    while start + segment_length <= total_sample_count:
        # Extract segment
        segment = audio[start:start + segment_length]
        segments.append(segment)
        # Move to next segment start
        start += segment_interval
    return segments

def resample_audio(audio, sr, target_sr, segment_length_seconds):
    """Downsample audio to target sample rate. Here I use librosa.resample, but feel free to use other methods.
    
    Args:
        audio: Input audio array (float32 in range -1 to 1)
        sr: Current sample rate
        target_sr: Target sample rate
    Returns:
        Downsampled audio array
    """
    output_audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)
    # output_audio = scipy_resample(audio, num=audio.shape[0] * target_sr / sr)
    # output_audio = resampy.resample(audio, sr, target_sr)
    # output_audio = soxr.resample(audio, sr, target_sr)

    # Possibly due to rounding error, the length of the output audio is not exactly the target length
    # trim or pad the audio to the target length
    output_audio = trim_or_pad(output_audio, target_sr, segment_length_seconds)

    return output_audio

