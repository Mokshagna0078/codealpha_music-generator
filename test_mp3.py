"""Quick test of notes_to_mp3 with a simple C major chord."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.midi_to_audio import notes_to_mp3

notes = [
    (0.0, 60, 80, 1.0),   # C4
    (0.0, 64, 80, 1.0),   # E4
    (0.0, 67, 80, 1.0),   # G4
    (1.0, 72, 80, 1.5),   # C5
]
out = Path("output/test_mp3.mp3")
out.parent.mkdir(parents=True, exist_ok=True)
notes_to_mp3(notes, out)
size_kb = out.stat().st_size / 1024
out.unlink()  # clean up
print(f"MP3 test OK — {size_kb:.0f} KB generated")
