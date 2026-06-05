"""Analyze generated MIDI files for musical statistics."""
from pathlib import Path
from utils.midi_utils import load_midi, extract_notes
from collections import Counter

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def note_name(pitch):
    return f"{NOTE_NAMES[(pitch - 21) % 12]}{pitch // 12 - 1}"

files = sorted(Path("output").glob("*.mid"))
print(f"Found {len(files)} MIDI files\n")

for f in files:
    midi = load_midi(f)
    notes = extract_notes(midi)
    if not notes:
        print(f"{f.name}: No notes found")
        continue

    start_times = [n[0] for n in notes]
    pitches = [n[1] for n in notes]
    velocities = [n[2] for n in notes]
    end_times = [n[3] for n in notes]
    durations = [e - s for s, e in zip(start_times, end_times)]

    total_dur = max(end_times)
    avg_pitch = sum(pitches) / len(pitches)
    avg_vel = sum(velocities) / len(velocities)
    min_pitch = min(pitches)
    max_pitch = max(pitches)
    avg_dur = sum(durations) / len(durations)
    notes_per_sec = len(notes) / total_dur if total_dur > 0 else 0
    unique_pitches = sorted(set(pitches))
    pitch_counts = Counter(pitches)
    most_common_pitch, most_common_count = pitch_counts.most_common(1)[0]

    sep = "=" * 45
    print(sep)
    print(f"  {f.name}")
    print(sep)
    print(f"  Notes:          {len(notes):4d}")
    print(f"  Duration:       {total_dur:6.1f}s")
    print(f"  Note density:   {notes_per_sec:5.2f} notes/sec")
    print(f"  Tempo:          120 BPM")
    print(f"  Instrument:     Acoustic Grand Piano")
    print()
    print(f"  Pitch range:    {min_pitch} ({note_name(min_pitch):>4}) -> {max_pitch} ({note_name(max_pitch):>4})")
    print(f"  Avg pitch:      {avg_pitch:5.1f} ({note_name(int(avg_pitch)):>4})")
    print(f"  Most used:      {most_common_pitch} ({note_name(most_common_pitch):>4}) - {most_common_count}x")
    print(f"  Unique pitches: {len(unique_pitches):4d}")
    print()
    print(f"  Avg velocity:   {avg_vel:5.1f} / 127")
    print(f"  Avg note dur:   {avg_dur:5.2f}s")
    print(f"  Max velocity:   {max(velocities):3d}")
    print(f"  Min velocity:   {min(velocities):3d}")
    print(f"  Max duration:   {max(durations):5.2f}s")
    print(f"  Min duration:   {min(durations):5.3f}s")
    print()

    pitch_span = max_pitch - min_pitch
    if pitch_span > 40:
        print(f"  Style hint:     Wide range - romantic/bold")
    elif pitch_span > 25:
        print(f"  Style hint:     Moderate range - classical")
    else:
        print(f"  Style hint:     Narrow range - minimal/ambient")

    if avg_vel > 100:
        print(f"  Dynamics:       Loud, forceful")
    elif avg_vel > 70:
        print(f"  Dynamics:       Moderate")
    else:
        print(f"  Dynamics:       Soft, gentle")

    if notes_per_sec > 8:
        print(f"  Rhythm:         Fast, dense (e.g., 16th notes)")
    elif notes_per_sec > 4:
        print(f"  Rhythm:         Moderate (e.g., 8th notes)")
    else:
        print(f"  Rhythm:         Slow, spacious")

    print(sep)
    print()
