from pathlib import Path
import yaml
import os
import librosa
from Utils_preprocessing import combine_audios, trim_or_pad, normalize_dynamics, stereo_to_mono
from Utils_audio_inspector import inspect_audio
import shutil # Shell utilities for copying files
import soundfile as sf

root_dir = Path(__file__).parent.parent.parent
midi_dir = root_dir / "datasets/1_midi_generated/raw/midi"
raw_audio_dir = root_dir / "datasets/1_midi_generated/raw/raw_audio"
target_dir = root_dir / "datasets/1_midi_generated/audio_dataset"

def load_config():
    config_path = root_dir / "conf" / "datasets" / "1_raw_to_training.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    process_list = range(config['raw_to_training']['start_index'], config['raw_to_training']['end_index'])
    # Collect raw audio file directories
    raw_audio_directories = [str(name) for name in process_list
                        if os.path.isdir(os.path.join(raw_audio_dir, str(name)))]
    sample_rates = config['sample_rates']
    total_files = len(raw_audio_directories) * len(sample_rates)
    processed_count = 0
    error_count = 0
    error_file_name = []

    for sample_rate in sample_rates:
        for raw_audio_file_number in raw_audio_directories:
            directory = os.path.join(raw_audio_dir, raw_audio_file_number, f"{sample_rate}Hz")
            output_directory = os.path.join(target_dir, f"{sample_rate}Hz", raw_audio_file_number)
            os.makedirs(output_directory, exist_ok=True)

            # Construct the path to the raw audio files, metadata files and the output files
            metadata_file_path = os.path.join(midi_dir, raw_audio_file_number, f"{raw_audio_file_number}_metadata.txt")
            raw_audio_file_path_piano = os.path.join(directory, f"{raw_audio_file_number}_piano_{sample_rate}Hz.wav")
            raw_audio_file_path_violin = os.path.join(directory, f"{raw_audio_file_number}_violin_{sample_rate}Hz.wav")
            training_file_path_metadata = os.path.join(output_directory, f"{raw_audio_file_number}_metadata.txt")
            training_file_path_mix = os.path.join(output_directory, f"{raw_audio_file_number}_mixed.wav")
            training_file_path_piano = os.path.join(output_directory, f"{raw_audio_file_number}_piano.wav")
            training_file_path_violin = os.path.join(output_directory, f"{raw_audio_file_number}_violin.wav")
            
            # Check if the raw audio file exists
            if os.path.isfile(raw_audio_file_path_piano) and os.path.isfile(raw_audio_file_path_violin):
                print(f"Processing {raw_audio_file_number} with sample rate {sample_rate}Hz, {processed_count}/{total_files} files finished.")
                # Load raw piano and violin audio (in format of float32)
                raw_piano = librosa.load(raw_audio_file_path_piano, sr=sample_rate)[0]
                raw_violin = librosa.load(raw_audio_file_path_violin, sr=sample_rate)[0]
                # Convert to mono if needed
                if config['output_mono']:
                    raw_piano = stereo_to_mono(raw_piano)
                    raw_violin = stereo_to_mono(raw_violin)
                # Trim or pad to target duration
                raw_piano = trim_or_pad(raw_piano, sample_rate, target_duration=config['target_duration'], silence_threshold=config['silence_threshold'])
                raw_violin = trim_or_pad(raw_violin, sample_rate, target_duration=config['target_duration'], silence_threshold=config['silence_threshold'])
                # Normalize dynamics
                piano = normalize_dynamics(raw_piano, target_db=config['target_db'], threshold_db=config['threshold_db'])
                violin = normalize_dynamics(raw_violin, target_db=config['target_db'], threshold_db=config['threshold_db'])
                # Mix audio
                mix = combine_audios(raw_piano, raw_violin, sample_rate)
                # Normalize the dynamics of the mix
                mix = normalize_dynamics(mix, target_db=config['target_db'], threshold_db=config['threshold_db'])
                # Save audio
                sf.write(training_file_path_mix, mix, sample_rate)
                sf.write(training_file_path_piano, piano, sample_rate)
                sf.write(training_file_path_violin, violin, sample_rate)
                # Inspect the audio
                sample_rate, bit_depth, duration, num_frames, rms_dBFS, peak_dBFS, LUFS = inspect_audio(training_file_path_mix, print_to_console=False)
                # Save metadata
                with open(metadata_file_path, 'r') as source:
                    with open(training_file_path_metadata, 'w') as dest:
                        dest.write(source.read())
                        dest.write(f"\nSample rate: {sample_rate}Hz")
                        dest.write(f"\nBit depth: {bit_depth} bits")
                        dest.write(f"\nDuration: {duration:.2f} seconds")
                        dest.write(f"\nNumber of frames: {num_frames}")
                        dest.write(f"\nRMS dBFS: {rms_dBFS:.1f} dB")
                        dest.write(f"\nPeak dBFS: {peak_dBFS:.1f} dB")
                        dest.write(f"\nLUFS: {LUFS:.1f} dB")
            else:
                print(f"Raw audio file(s) not found: {raw_audio_file_number}")
                error_count += 1
                error_file_name.append(raw_audio_file_number)

            processed_count += 1

    print(f"Processing complete. {processed_count}/{total_files} files processed successfully.")

if __name__ == "__main__":
    main()
