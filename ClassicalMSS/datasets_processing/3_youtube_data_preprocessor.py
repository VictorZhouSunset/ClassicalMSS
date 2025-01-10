"""
As a primary way of evaluational dataset, we download some piano solo and violin solo performances from YouTube and manually mix them together.
This script is used to create this dataset. The steps are as follows:
1. Segment the audio files into segments, only keep the segments with low silence proportion.
2. Resample the segments to the target sample rate.
3. Normalize the dynamics of the segments.
4. Mix the segments (temporarily, one segment used for only one time).
** In this case, from the 12 songs downloaded from YouTube, we got 736 segments of 5 seconds in this test dataset,
** if we allow a segment to be used more than one time, we can get at most 736 * 736 = 541696 segments.
5. Normalize the dynamics of the mix.
6. Save the dataset (piano_segment, violin_segment, mixed_segment) in the target folder.
"""

import yaml
from pathlib import Path
from Utils_preprocessing import get_silence_proportion, stereo_to_mono, segment_audio, resample_audio, normalize_dynamics
import soundfile as sf

root_dir = Path(__file__).parent.parent.parent
input_pianos_dir = root_dir / "datasets" / "3_youtube_test" / "raw" / "YouTube_original" / "piano"
input_violins_dir = root_dir / "datasets" / "3_youtube_test" / "raw" / "YouTube_original" / "violin"
output_dir = root_dir / "datasets" / "3_youtube_test" / "audio_dataset"

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def batch_segment_audio(input_dir, config):
    """
    Segment all audio files in a director, return a list of qualified segments and the sample rate of the audio files.
    """
    qualified_segments = []
    for file in input_dir.iterdir():
        if file.is_file() and (file.suffix == ".wav" or file.suffix == ".mp3" or file.suffix == ".m4a" or file.suffix == ".ogg" or file.suffix == ".flac"):
            audio, current_sample_rate = sf.read(file)
            segment_length_seconds = config['segmentation']['segment_length_seconds']
            segment_window_interval_seconds = config['segmentation']['segment_window_interval_seconds']
            output_mono = config['output_mono']
            target_sr = config['target_sr']
            segments = segment_audio(audio, current_sample_rate, segment_length_seconds=segment_length_seconds, segment_window_interval_seconds=segment_window_interval_seconds, output_mono=output_mono)
            for segment in segments:
                if get_silence_proportion(segment, threshold_db=config['silence_threshold_db']) < config['max_silence_proportion']:
                    qualified_segments.append(segment)
    return qualified_segments, current_sample_rate

def main():
    config = load_config(root_dir / "conf" / "datasets" / "3_youtube_data_preprocessor.yaml")
    normalizing_config = load_config(root_dir / "conf" / "datasets" / "1_raw_to_training.yaml")
    segment_length_seconds = config['segmentation']['segment_length_seconds']
    
    qualified_piano_segments, piano_sample_rate = batch_segment_audio(input_pianos_dir, config)
    qualified_violin_segments, violin_sample_rate = batch_segment_audio(input_violins_dir, config)
    training_count = min(len(qualified_piano_segments), len(qualified_violin_segments))
    # Trim the longer list to match the shorter one
    qualified_piano_segments = qualified_piano_segments[:training_count]
    qualified_violin_segments = qualified_violin_segments[:training_count]

    for sr in config['target_sr']:
        print(f"Processing {sr}Hz segments...")
        # resample the segments to the target sample rate
        if sr != piano_sample_rate:
            piano_segments = [resample_audio(segment, piano_sample_rate, sr, segment_length_seconds) for segment in qualified_piano_segments]
        else:
            piano_segments = qualified_piano_segments
        
        if sr != violin_sample_rate:
            violin_segments = [resample_audio(segment, violin_sample_rate, sr, segment_length_seconds) for segment in qualified_violin_segments]
        else:
            violin_segments = qualified_violin_segments
        # normalize the dynamics of the segments (training_count segments)
        piano_segments = [normalize_dynamics(segment, normalizing_config['target_db'], normalizing_config['threshold_db']) for segment in piano_segments]
        violin_segments = [normalize_dynamics(segment, normalizing_config['target_db'], normalizing_config['threshold_db']) for segment in violin_segments]
        # mix the segments (training_count segments, no crossing)
        mixed_segments = [(piano_segments[i] + violin_segments[i]) / 2 for i in range(training_count)]
        # normalize the dynamics of the mix (training_count segments)
        mixed_segments = [normalize_dynamics(segment, normalizing_config['target_db'], normalizing_config['threshold_db']) for segment in mixed_segments]
        # save the dataset in sr folder (training_count segments, sr)
        output_dir_sr = output_dir / f"{sr}Hz"
        output_dir_sr.mkdir(parents=True, exist_ok=True)
        for i, segment in enumerate(mixed_segments):
            if i % 100 == 0:
                print(f"Processing {i}th segment...")
            output_segment_dir = output_dir_sr / f"{i}"
            output_segment_dir.mkdir(parents=True, exist_ok=True)
            sf.write(output_segment_dir / f"{i}_mixed.wav", segment, sr)
            sf.write(output_segment_dir / f"{i}_piano.wav", piano_segments[i], sr)
            sf.write(output_segment_dir / f"{i}_violin.wav", violin_segments[i], sr)

if __name__ == "__main__":
    main()

