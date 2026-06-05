"""Convert all existing MIDI files in output/ to MP3."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.midi_to_audio import midi_file_to_mp3

out_dir = Path("output")
midi_files = sorted(out_dir.glob("*.mid"))
print(f"Found {len(midi_files)} MIDI files")

for f in midi_files:
    mp3 = f.with_suffix(".mp3")
    print(f"  Converting {f.name} -> {mp3.name} ...")
    midi_file_to_mp3(f, mp3)
    size_kb = mp3.stat().st_size / 1024
    print(f"    Done: {mp3.name} ({size_kb:.0f} KB)")

print("\nAll done! Double-click any .mp3 file to play.")
