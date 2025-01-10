"""
This script renders MIDI files to WAV files using FluidSynth (batch processing the command line), make sure the FluidSynth is installed and accessible.
The soundfonts may be under copyright, so you may need to download your own.
FluidSynth is available at https://www.fluidsynth.org/
** There is no direct bit depth control in FluidSynth, the bit depth is set to 16 on my computer.
** The generated wav files are mono
Soundfonts used in this script are:
    -- FluidR3_GM, available at https://member.keymusician.com/Member/FluidR3_GM/index.html
"""

import os
import subprocess
import yaml
from pathlib import Path

def load_config():
    config_path = Path(__file__).parent.parent.parent / "conf" / "datasets" / "1_midi_renderer.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def build_fluidsynth_command(midi_file_path, wav_file_path, soundfont_path, sample_rate=48000, buffer_size=512):
    return [
        fluidsynth_executable,
        '-ni',  # Disable MIDI input and interactive mode
        '-F', wav_file_path,  # Output file
        '-r', str(sample_rate),
        '-z', str(buffer_size),
        soundfont_path,
        midi_file_path
    ]

def run_fluidsynth_command(midi_file_number, command, sample_rate):
    global processed_count, error_count
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processed_count += 1
    except subprocess.CalledProcessError as e:
        error_count += 1
        error_message = e.stderr.decode()
        print(f"Error processing {midi_file_number} at {sample_rate} Hz: {error_message}")

def main():
    global processed_count, error_count, fluidsynth_executable
    
    # Initialize global variables
    processed_count = 0
    error_count = 0
    error_file_name = []
    fluidsynth_executable = 'fluidsynth'  # Or provide the full path to fluidsynth.exe

    # Path to the base directory containing the MIDI files
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base_directory = os.path.join(project_root, "datasets", "1_midi_generated", "raw", "midi")
    os.makedirs(base_directory, exist_ok=True)

    config = load_config()
    # Path to your SoundFont file
    if config['soundfont_name'] == "FluidR3_GM":
        soundfont_path = os.path.join(project_root, "ClassicalMSS", "soundfonts", "FluidR3_GM", "FluidR3_GM.sf2")
    else:
        raise ValueError(f"Soundfont {config['soundfont_name']} not found")

    process_list = range(config['midi_to_render']['start_index'], config['midi_to_render']['end_index'])
    # Collect MIDI file directories
    midi_directories = [str(name) for name in process_list
                        if os.path.isdir(os.path.join(base_directory, str(name)))]

    sample_rates = config['sample_rates']
    total_files = len(midi_directories) * 2 * len(sample_rates)

d    for midi_file_number in midi_directories:
        directory = os.path.join(base_directory, midi_file_number)
        output_directory = os.path.join(project_root, "datasets", "1_midi_generated", "raw", "raw_audio", midi_file_number)
        os.makedirs(output_directory, exist_ok=True)
        
        # Construct the path to the MIDI file
        midi_file_path_piano = os.path.join(directory, f"{midi_file_number}_piano.mid")
        midi_file_path_violin = os.path.join(directory, f"{midi_file_number}_violin.mid")
        
        # Check if the MIDI file exists
        if os.path.isfile(midi_file_path_piano) and os.path.isfile(midi_file_path_violin):
            # Render at different sample rates
            for sample_rate in sample_rates:
                # Construct the output WAV file path
                sample_rate_directory = os.path.join(output_directory, f"{sample_rate}Hz")
                os.makedirs(sample_rate_directory, exist_ok=True)
                wav_file_path_piano = os.path.join(sample_rate_directory, f"{midi_file_number}_piano_{sample_rate}Hz.wav")
                wav_file_path_violin = os.path.join(sample_rate_directory, f"{midi_file_number}_violin_{sample_rate}Hz.wav")
                
                # Build and run the FluidSynth commands
                command_piano = build_fluidsynth_command(midi_file_path_piano, wav_file_path_piano, soundfont_path, sample_rate=sample_rate)
                command_violin = build_fluidsynth_command(midi_file_path_violin, wav_file_path_violin, soundfont_path, sample_rate=sample_rate)
                
                run_fluidsynth_command(midi_file_number, command_piano, sample_rate)
                print(f"Processing {midi_file_number}_piano at {sample_rate} Hz ({processed_count}/{total_files})...")
                run_fluidsynth_command(midi_file_number, command_violin, sample_rate)
                print(f"Processing {midi_file_number}_violin at {sample_rate} Hz ({processed_count}/{total_files})...")
        else:
            print(f"MIDI file(s) not found: {midi_file_number}")
            error_count += 1
            error_file_name.append(midi_file_number)

    print(f"\nProcessing complete. {processed_count}/{total_files} files processed successfully.")
    if error_count > 0:
        print(f"{error_count} files encountered errors.")
        print(f"Error file names: {error_file_name}")

if __name__ == "__main__":
    main()
