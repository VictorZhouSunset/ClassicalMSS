"""
The goal of this script is to generate a piano midi and a violin midi and they are correlated
so that when put together, they sound like a duet that makes some senses.
** What does it mean by "makes some senses" is debatable, here only reflects my choice.

Because of the limitation of the resources, the length of the midi is limited to about 5 seconds.
** This is a temporary limitation, and may be removed in the future.

The logic of the generation is as follows:

1. Basic properties of the segments:
    (1) key signature (selected from all major and minor keys)
    (2) time signature (selected from 3/4, 4/4, 5/4, with probability 45%, 45%, 10%)
    (3) tempo (quarter note = 72 - 144)

2. From the time signature and the tempo, the maximum number of complete bars in 5 seconds can be determined.
** Here, the number of bars will be "that maximum number + 1" to avoid long quiet sections.
** Later in the rendering process, the resulting audio will be trimmed to 5 seconds.

3. Randomly place the chord-change (potentially pedal) positions, accurate to one beat, until the end of the last bar;
Every two successive chord-change positions extend a "chord window". For each chord window:
    (1) Determine whether there is pedaled through or not (70% with pedal, 30% without pedal)
    (2) Randomly select the chord in the chord window.
        (a) Determine the chord:
        |    chord    |  I  |  II | III | IV  |  V  | VI  | VII |
        | probability | 28% | 10% | 5%  | 18% | 24% | 10% | 5%  |
        (b) Determine the chord type:
        | chord type  | 3: triad | 4: 7th chord | 5: 9th chord | 6: 11th chord | 7: 13th chord |
        | probability |    70%   |      20%     |      6%      |      3%       |     1%        |

4. Within each chord window, randomly select two piano pitch ranges, without overlap; and one violin pitch range (may overlap with piano);
** This prevents suspension notes and I am planning to alter the algorithm later.
For each pitch range, generate a melody inside the pitch range:
    (1) Randomly pick the notes durations until the end of the whole segment (may be legato to the next note);
    (2) According to which chord frame each note is in, randomly select the note pitch (may be rest) from the available notes;
    ** The available notes are the notes in the chord plus a small probability of notes outside of the chord;
    ** Notes near the previous note are more likely to be selected;
    (3) Randomly determine the note velocity;
    ** The velocity is more likely to be close to the previous velocity;

5. Write in the midi files (there will be two correlated midi files, one for piano and one for violin);

The randomness of the generation is controlled by the random seed in the config file.
** Only the "random" package is used in this script so if you want to add some randomness using other packages like numpy.random,
** make sure to set a random seed for them too to ensure reproducibility.
"""
from midiutil import MIDIFile
import random
import os
import time
import yaml
from pathlib import Path

def load_config():
    config_path = Path(__file__).parent.parent.parent / "conf" / "datasets" / "1_midi_generator.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Major and minor key mappings by accidentals (sharps or flats)
major_key_map = {
    -6: "Gb", -5: "Db", -4: "Ab", -3: "Eb", -2: "Bb", -1: "F",
    0: "C", 1: "G", 2: "D", 3: "A", 4: "E", 5: "B" 
}
minor_key_map = {
    -6:"Ebm",-5: "Bbm", -4: "Fm", -3: "Cm", -2: "Gm", -1: "Dm", 0: "Am",
    1: "Em", 2: "Bm", 3: "F#m", 4: "C#m", 5: "G#m"
}
# Define all major and harmonic minor scales from 6 flats to 5 sharps
scales = {
    'Gb': ['Gb', 'Ab', 'Bb', 'Cb', 'Db', 'Eb', 'F'],
    'Db': ['Db', 'Eb', 'F', 'Gb', 'Ab', 'Bb', 'C'],
    'Ab': ['Ab', 'Bb', 'C', 'Db', 'Eb', 'F', 'G'],
    'Eb': ['Eb', 'F', 'G', 'Ab', 'Bb', 'C', 'D'],
    'Bb': ['Bb', 'C', 'D', 'Eb', 'F', 'G', 'A'],
    'F': ['F', 'G', 'A', 'Bb', 'C', 'D', 'E'],
    'C': ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
    'G': ['G', 'A', 'B', 'C', 'D', 'E', 'F#'],
    'D': ['D', 'E', 'F#', 'G', 'A', 'B', 'C#'],
    'A': ['A', 'B', 'C#', 'D', 'E', 'F#', 'G#'],
    'E': ['E', 'F#', 'G#', 'A', 'B', 'C#', 'D#'],
    'B': ['B', 'C#', 'D#', 'E', 'F#', 'G#', 'A#'],
    'Ebm': ['Eb', 'F', 'Gb', 'Ab', 'Bb', 'Cb', 'D'],
    'Bbm': ['Bb', 'C', 'Db', 'Eb', 'F', 'Gb', 'A'],
    'Fm': ['F', 'G', 'Ab', 'Bb', 'C', 'Db', 'E'],
    'Cm': ['C', 'D', 'Eb', 'F', 'G', 'Ab', 'B'],
    'Gm': ['G', 'A', 'Bb', 'C', 'D', 'Eb', 'F#'],
    'Dm': ['D', 'E', 'F', 'G', 'A', 'Bb', 'C#'],
    'Am': ['A', 'B', 'C', 'D', 'E', 'F', 'G#'],
    'Em': ['E', 'F#', 'G', 'A', 'B', 'C', 'D#'],
    'Bm': ['B', 'C#', 'D', 'E', 'F#', 'G', 'A#'],
    'F#m': ['F#', 'G#', 'A', 'B', 'C#', 'D', 'E#'],
    'C#m': ['C#', 'D#', 'E', 'F#', 'G#', 'A', 'B#'],
    'G#m': ['G#', 'A#', 'B', 'C#', 'D#', 'E', 'F#']
}
# Map of notes'MIDI numbers to their remainder when divided by 12
notes_remainder_map = {
   "B#": 0, "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "Fb": 4, "E#": 5, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11
}


def select_metadata(config):
    # Randomly select tempo, key signature, and time signature
    selected_tempo = random.randint(config['tempo']['min'], config['tempo']['max'])
    # Randomly select mode (0 for major, 1 for minor)
    selected_mode = random.choice([0, 1])
    # Randomly select the number of accidentals (-6 to 5)
    selected_accidentals = random.randint(-6, 5)
    # Determine the accidental type based on the number of accidentals, 1 for flat keys, 0 for sharp keys
    selected_accidental_types = 1 if selected_accidentals < 0 else 0
    # Determine the key based on the mode and number of accidentals
    selected_key = major_key_map[selected_accidentals] if selected_mode == 0 else minor_key_map[selected_accidentals]
    # Randomly select the time signature
    selected_time_signature = random.choices(config['time_signatures']['options'], weights=config['time_signatures']['weights'])[0]
    return selected_tempo, selected_mode, selected_accidentals, selected_accidental_types, selected_key, selected_time_signature

def find_max_bars(tempo, time_signature, config):
    max_seconds = config['duration']['max_seconds']
    beats_per_minute = tempo
    beats_per_second = beats_per_minute / 60
    beats_per_bar = time_signature[0]
    seconds_per_bar = beats_per_bar / beats_per_second
    max_bars = int(max_seconds / seconds_per_bar) + 1 # To avoid long quiet sections
    max_beats = max_bars * beats_per_bar
    return max_bars, max_beats

def create_chord_window_tuples(total_beats, config):
    # Create a list of chord windows
    chord_windows = []
    
    # Define chord probabilities
    # 1: I, 2: II, 3: III, 4: IV, 5: V, 6: VI, 7: VII
    chord_probs = config['chord_probabilities']
    chords = list(range(1, 8))  # [1-7] for chord degrees
    chord_weights = [chord_probs[degree] for degree in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']]

    # Define chord type probabilities
    # 3: Triad, 4: 7th chord, 5: 9th chord, 6: 11th chord, 7: 13th chord
    type_probs = config['chord_type_probabilities']
    types = [3, 4, 5, 6, 7]  # Corresponding to triad through thirteenth
    type_weights = [type_probs[t] for t in ['triad', 'seventh', 'ninth', 'eleventh', 'thirteenth']]

    pedal_prob = config['pedal']['probability']

    # Create a list to store the start beats of chord windows
    chord_window_starts = [0]  # The first beat is always a start
    for beat in range(1, total_beats):
        if random.random() < config['chord_window_start_probability']:
            chord_window_starts.append(beat)

    for i, beat in enumerate(chord_window_starts):
        next_beat = chord_window_starts[i + 1] if i + 1 < len(chord_window_starts) else total_beats
        duration = next_beat - beat
        chord = random.choices(chords, weights=chord_weights)[0]
        chord_type = random.choices(types, weights=type_weights)[0]
        pedal = random.choices([True, False], weights=[pedal_prob, 1-pedal_prob])[0]
        chord_windows.append((beat, duration, chord, chord_type, pedal))
            
    return chord_windows

def find_notes_in_chord(key, chord, chord_type):
    notes_in_key = scales[key]
    root_note = notes_in_key[chord-1]
    third = notes_in_key[(chord+2)%7]
    fifth = notes_in_key[(chord+4)%7]
    seventh = notes_in_key[(chord+6)%7]
    ninth = notes_in_key[(chord+8)%7]
    eleventh = notes_in_key[(chord+10)%7]
    thirteenth = notes_in_key[(chord+12)%7]

    if chord_type == 3:
        notes_in_chord = [root_note, third, fifth]
    elif chord_type == 4:
        notes_in_chord = [root_note, third, fifth, seventh]
    elif chord_type == 5:
        notes_in_chord = [root_note, third, fifth, seventh, ninth]
    elif chord_type == 6:
        notes_in_chord = [root_note, third, fifth, seventh, ninth, eleventh]
    elif chord_type == 7:
        notes_in_chord = [root_note, third, fifth, seventh, ninth, eleventh, thirteenth]

    return notes_in_chord

def get_usable_notes(voice_range, notes_to_choose_from):
    notes_remainders_to_choose_from = [notes_remainder_map[note] for note in notes_to_choose_from]
    usable_notes = []
    for midi_note_number in voice_range:
        if midi_note_number % 12 in notes_remainders_to_choose_from:
            usable_notes.append(midi_note_number)
    # Convert usable_notes to a set to remove duplicates
    usable_notes = set(usable_notes)
    return usable_notes
def get_note_durations(duration, config):
    available_durations = config['available_durations']
    remaining_duration = duration
    note_durations = []
    while remaining_duration > 0:
        possible_durations = [d for d in available_durations if d <= remaining_duration]
        if not possible_durations: # If there's no possible durations, break the loop
            break
        chosen_duration = random.choice(possible_durations)
        note_durations.append(chosen_duration)
        remaining_duration -= chosen_duration
    # If there's any remaining duration, add it as the smallest possible note
    if remaining_duration > 0:
        note_durations.append(remaining_duration)
    return note_durations

def generate_melody_for_one_voice(midi, track, channel, beat, duration, notes_to_choose_from, voice_range, config, rest_poss=0.2, legato_poss=0.5, melody_temp = 4, stick_to_chord = 3, velocity_temp = 4):
    """
    The melody_temp and velocity_temp determine how much probability of choosing a note close to the previous note.
    The stick_to_chord determines how much access probability of choosing a note in the chord notes.
    """
    # Get the notes that are in the voice range and in the chord given
    usable_notes = get_usable_notes(voice_range, notes_to_choose_from)    
    # Divide the duration into a composition of note durations
    note_durations = get_note_durations(duration, config)
    curser = beat
    last_note = None
    last_velocity = None
    new_usable_notes = usable_notes.copy()
    for current_note_duration in note_durations:
        # There is a rest_poss probability that the note will be a rest
        if random.random() < rest_poss:
            curser += current_note_duration
        else:
            # If legato is true, the note will be a bit longer than the intended duration
            legato = random.random() < legato_poss
            # If this is the first note, choose a random note from the usable_notes
            if not last_note:
                chosen_note = random.choice(tuple(new_usable_notes))
                chosen_velocity = random.randint(config['first_note_velocity']['min'], config['first_note_velocity']['max']) # both inclusive
            else:
                # Calculate weights based on distance from last_note, the smaller the distance, the higher the weight
                # The weight is also higher if the note is in the original usable_notes (chord notes)
                weights = [
                    (1 / (abs(note - last_note) + melody_temp)) * (stick_to_chord if note in usable_notes else 1) 
                    for note in new_usable_notes
                ]
                chosen_note = random.choices(tuple(new_usable_notes), weights=weights, k=1)[0]
                if not last_velocity:
                    chosen_velocity = random.randint(config['first_note_velocity']['min'], config['first_note_velocity']['max'])
                else:
                    usable_velocities = range(max(config['velocity_range']['min'], last_velocity-15), min(config['velocity_range']['max'], last_velocity+15) + 1)
                    velocity_weights = [1 / (abs(velocity - last_velocity) + velocity_temp) for velocity in usable_velocities]
                    chosen_velocity = random.choices(usable_velocities, weights=velocity_weights, k=1)[0]
            if legato:
                actual_note_duration = current_note_duration + config['melody']['legato_level']
            else:
                actual_note_duration = current_note_duration
            midi.addNote(track, channel, chosen_note, curser, actual_note_duration, chosen_velocity)
            curser += current_note_duration
            last_note = chosen_note
            last_velocity = chosen_velocity
            new_usable_notes = usable_notes.copy() | set(range(chosen_note - 2, chosen_note + 3))

def add_pedal_to_midi(midi, track, channel, beat, duration, config, pedal=False):
    # pedal_biase is the time for the syncopated pedaling
    pedal_biase = config['pedal']['bias']
    if pedal:
        midi.addControllerEvent(track, channel, beat + pedal_biase, 64, 127)
        midi.addControllerEvent(track, channel, beat + duration + 0.8*pedal_biase, 64, 0)
def create_voice_range(note_range, voice_range_width = 24):
    # For the piano, it is to generate a random start note between 21 and 85 (108 - 24 + 1), inclusive
    start_note = random.randint(note_range[0], note_range[1] - voice_range_width + 1)
    # Create a range of two octaves (24 semitones) from the start note
    voice_range = range(start_note, start_note + voice_range_width)
    return voice_range


def create_midi_violin_piano(midi_file_name, directory, config, verbose=False):
    # Create a MIDIFile object with 2 tracks
    midi_piano = MIDIFile(1)
    midi_violin = MIDIFile(1)

    # Set the track names
    midi_piano.addTrackName(0, 0, "Piano")
    midi_violin.addTrackName(0, 0, "Violin")
    
    # Set instruments
    midi_piano.addProgramChange(0, 0, 0, config['instruments']['piano']['program'])  # Piano
    midi_violin.addProgramChange(0, 0, 0, config['instruments']['violin']['program'])  # Violin

    # Call the function to get the selected parameters
    selected_tempo, selected_mode, selected_accidentals, selected_accidental_types, selected_key, selected_time_signature = select_metadata(config)

    # Set tempo for both tracks
    midi_piano.addTempo(0, 0, selected_tempo)
    midi_violin.addTempo(0, 0, selected_tempo)

    # Set key signature for both tracks
    midi_piano.addKeySignature(0, 0, selected_accidentals, selected_accidental_types, selected_mode)
    midi_violin.addKeySignature(0, 0, selected_accidentals, selected_accidental_types, selected_mode)

    # Set time signature for both tracks
    midi_piano.addTimeSignature(0, 0, selected_time_signature[0], selected_time_signature[1], 24, 8)
    midi_violin.addTimeSignature(0, 0, selected_time_signature[0], selected_time_signature[1], 24, 8)

    # Calculate the number of bars that can fit in a set-time window
    total_bars, total_beats = find_max_bars(selected_tempo, selected_time_signature, config)

    note_range_piano = config['instruments']['piano']['range']
    note_range_violin = config['instruments']['violin']['range']
    piano_voice_1_range = create_voice_range(note_range_piano)
    piano_voice_2_range = create_voice_range(note_range_piano)
    violin_voice_range = create_voice_range(note_range_violin)
    # Create chord windows
    chord_windows = create_chord_window_tuples(total_beats, config) # List of tuples (beat, duration, chord, chord_type, pedal)
    for window in chord_windows:
        beat, duration, chord, chord_type, pedal = window
        # Create only one pedal marking for piano sound only
        add_pedal_to_midi(midi_piano, 0, 0, beat, duration, config, pedal=pedal)
        notes_to_choose_from = find_notes_in_chord(selected_key, chord, chord_type)
        
        generate_melody_for_one_voice(midi_piano, 0, 0, beat, duration, notes_to_choose_from, piano_voice_1_range, config, rest_poss=config['instruments']['piano']['rest_poss'], legato_poss=config['instruments']['piano']['legato_poss'], melody_temp=config['melody']['pitch_temp'], stick_to_chord=config['melody']['stick_to_chord'], velocity_temp=config['melody']['velocity_temp'])
        generate_melody_for_one_voice(midi_piano, 0, 0, beat, duration, notes_to_choose_from, piano_voice_2_range, config, rest_poss=config['instruments']['piano']['rest_poss'], legato_poss=config['instruments']['piano']['legato_poss'], melody_temp=config['melody']['pitch_temp'], stick_to_chord=config['melody']['stick_to_chord'], velocity_temp=config['melody']['velocity_temp'])
        generate_melody_for_one_voice(midi_violin, 0, 0, beat, duration, notes_to_choose_from, violin_voice_range, config, rest_poss=config['instruments']['violin']['rest_poss'], legato_poss=config['instruments']['violin']['legato_poss'], melody_temp=config['melody']['pitch_temp'], stick_to_chord=config['melody']['stick_to_chord'], velocity_temp=config['melody']['velocity_temp'])

    # Save MIDI file
    with open(os.path.join(directory, f"{midi_file_name}_piano.mid"), "wb") as output_file:
        midi_piano.writeFile(output_file)
    with open(os.path.join(directory, f"{midi_file_name}_violin.mid"), "wb") as output_file:
        midi_violin.writeFile(output_file)

    # Save metadata to a text file
    metadata_filename = os.path.join(directory, f"{midi_file_name}_metadata.txt")
    with open(metadata_filename, "w") as metadata_file:
        metadata_file.write(f"Tempo: {selected_tempo} BPM\n")
        metadata_file.write(f"Key: {selected_key} (Mode: {'Major' if selected_mode == 0 else 'Minor'}, Accidentals: {selected_accidentals}, Accidental Types: {'Flat' if selected_accidental_types == 1 else 'Sharp'})\n")
        metadata_file.write(f"Time Signature: {selected_time_signature[0]}/{selected_time_signature[1]}\n")
        metadata_file.write(f"Number of bars: {total_bars}\n")
        metadata_file.write(f"Number of beats: {total_beats}\n")
    if verbose:
        print(midi_file_name + ".mid" + " saved")
        # Print the metadata
        print(f"Tempo: {selected_tempo} BPM")
        print(f"Key: {selected_key} (Mode: {'Major' if selected_mode == 0 else 'Minor'}, Accidentals: {selected_accidentals}, Accidental Types: {'Flat' if selected_accidental_types == 1 else 'Sharp'})")
        print(f"Time Signature: {selected_time_signature[0]}/{selected_time_signature[1]}")
        print(f"Number of bars: {total_bars}")
        print(f"Number of beats: {total_beats}")
    return midi_piano, midi_violin

# Main loop
def main():
    # Get the absolute path to the project root directory (2 levels up from current file)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Create the full path to the midi directory
    directory = os.path.join(project_root, "datasets", "1_midi_generated", "raw", "midi")
    os.makedirs(directory, exist_ok=True)

    config = load_config()
    random.seed(config['random_seed']) # Set the random seed to ensure reproducibility

    error_file_name = []
    for i in range(0, config['number_of_midi_generations']):
        attempts = 0
        max_attempts = config['max_attempts']  # Maximum number of attempts per index
        current_directory = os.path.join(directory, str(i))
        os.makedirs(current_directory, exist_ok=True)
        while attempts < max_attempts:
            try:
                create_midi_violin_piano(str(i), current_directory, config, verbose=False)
                if i % 100 == 0:
                    print(f"Successfully created MIDI file {i}")
                break
            except Exception as e:
                # There will be a "IndexError: pop from empty list" which seems to be related with the note-on/note-off events
                # I just ignore it here and try again
                attempts += 1
                # print(f"Error creating MIDI file {i}, attempt {attempts}: {str(e)}")
                if attempts == max_attempts:
                    print(f"Failed to create MIDI file {i} after {max_attempts} attempts. Moving to next index.")
                    error_file_name.append(i)
        # Optional: add a small delay to avoid potential issues with file writing
        time.sleep(0.1)

    print("MIDI generation complete.")
    if error_file_name:
        print(f"Failed to generate files for indices: {error_file_name}")

if __name__ == "__main__":
    main()