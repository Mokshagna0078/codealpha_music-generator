"""
MIDI utility functions for music generation.

Provides conversion between internal NoteEvent representation and MIDI files.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pretty_midi

# NoteEvent: (start_time, pitch, velocity, end_time)
NoteEvent = Tuple[float, int, int, float]


def notes_to_midi(
    notes: List[NoteEvent],
    tempo: float = 120.0,
    program: int = 0,  # Acoustic Grand Piano
) -> pretty_midi.PrettyMIDI:
    """
    Convert a list of NoteEvent tuples to a PrettyMIDI object.

    Args:
        notes: List of (start_time, pitch, velocity, end_time) tuples
        tempo: BPM
        program: MIDI program number (instrument)

    Returns:
        PrettyMIDI object
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=program)

    for start, pitch, velocity, end in notes:
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start,
            end=end,
        )
        piano.notes.append(note)

    midi.instruments.append(piano)
    return midi


def save_midi(midi: pretty_midi.PrettyMIDI, output_path: str | Path) -> None:
    """Save a PrettyMIDI object to a file."""
    midi.write(str(output_path))


def load_midi(file_path: str | Path) -> Optional[pretty_midi.PrettyMIDI]:
    """
    Load a MIDI file and return a PrettyMIDI object.

    Args:
        file_path: Path to the MIDI file

    Returns:
        PrettyMIDI object or None if loading fails
    """
    try:
        return pretty_midi.PrettyMIDI(str(file_path))
    except Exception as e:
        print(f"Error loading MIDI {file_path}: {e}")
        return None


def extract_notes(midi: pretty_midi.PrettyMIDI) -> List[NoteEvent]:
    """
    Extract NoteEvent tuples from a PrettyMIDI object.

    Returns:
        List of (start_time, pitch, velocity, end_time) tuples
    """
    notes: List[NoteEvent] = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            notes.append((note.start, note.pitch, note.velocity, note.end))
    return notes


def midi_file_to_notes(file_path: str | Path) -> List[NoteEvent]:
    """Load a MIDI file and extract notes directly."""
    midi = load_midi(file_path)
    if midi is None:
        return []
    return extract_notes(midi)
