"""
Convert MIDI note events to playable audio (WAV and MP3).

Uses pure-Python additive synthesis for WAV output, and pydub + ffmpeg
for MP3 conversion (much smaller file sizes).
"""
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.io import wavfile

from utils.midi_utils import NoteEvent, midi_file_to_notes

# ── Constants ─────────────────────────────────────────────────────────
SAMPLE_RATE = 44100
MAX_AMPLITUDE = 32767  # 16-bit audio


# ── Synthesis helpers ─────────────────────────────────────────────────

def _piano_harmonics(frequency: float, t: np.ndarray) -> np.ndarray:
    """
    Generate a piano-like waveform using harmonic synthesis.
    
    Each harmonic has decreasing amplitude to simulate the
    natural tone of a piano.
    """
    harmonics = [
        (1.0, 1.0),   # fundamental
        (0.5, 0.6),   # 2nd harmonic
        (0.3, 0.3),   # 3rd
        (0.2, 0.15),  # 4th
        (0.1, 0.08),  # 5th
        (0.05, 0.04), # 6th
    ]
    waveform = np.zeros_like(t)
    for ratio, gain in harmonics:
        waveform += gain * np.sin(2 * np.pi * frequency * ratio * t)
    return waveform / sum(g for _, g in harmonics)


def _apply_envelope(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Apply an ADSR-like amplitude envelope to simulate a piano note.
    
    - Attack: quick rise (~5ms)
    - Decay: slight drop (~20ms)
    - Sustain: gradual fade
    - Release: natural tail
    """
    n = len(waveform)
    attack = min(int(0.005 * sample_rate), n // 4)
    decay = min(int(0.02 * sample_rate), n // 4)

    envelope = np.ones(n)

    # Attack
    if attack > 0:
        envelope[:attack] = np.linspace(0, 1, attack)

    # Decay
    if decay > 0:
        end_decay = attack + decay
        if end_decay <= n:
            envelope[attack:end_decay] = np.linspace(1, 0.7, decay)

    # Release (smooth fade out)
    release_start = max(n - int(0.3 * sample_rate), end_decay)
    if release_start < n:
        envelope[release_start:] = np.linspace(
            envelope[release_start], 0, n - release_start
        )

    return waveform * envelope


def _note_to_samples(
    start_time: float,
    pitch: int,
    velocity: int,
    end_time: float,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Generate audio samples for a single note.
    
    Args:
        start_time: Note start in seconds
        pitch: MIDI pitch (0-127)
        velocity: MIDI velocity (0-127)
        end_time: Note end in seconds
        sample_rate: Audio sample rate
    
    Returns:
        Array of float samples
    """
    frequency = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
    duration = max(end_time - start_time, 0.05)  # minimum 50ms
    amplitude = (velocity / 127.0) * 0.5  # scale to avoid clipping

    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    waveform = _piano_harmonics(frequency, t)
    waveform = _apply_envelope(waveform, sample_rate)
    waveform *= amplitude

    return waveform


# ── Main API ──────────────────────────────────────────────────────────

def notes_to_wav(
    notes: List[NoteEvent],
    output_path: str | Path,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    """
    Convert a list of NoteEvents to a WAV file using additive synthesis.
    
    Args:
        notes: List of (start_time, pitch, velocity, end_time) tuples
        output_path: Path to save the WAV file
        sample_rate: Audio sample rate (default 44100)
    """
    if not notes:
        print("Warning: No notes to synthesize")
        return

    # Determine total duration
    max_end = max(n[3] for n in notes)
    total_samples = int(max_end * sample_rate) + sample_rate  # 1s padding

    # Mix all notes into a single buffer
    mix = np.zeros(total_samples, dtype=np.float64)

    for start, pitch, velocity, end in notes:
        samples = _note_to_samples(start, pitch, velocity, end, sample_rate)
        offset = int(start * sample_rate)

        # Pad or trim to fit in the mix buffer
        if offset + len(samples) > total_samples:
            samples = samples[: total_samples - offset]

        mix[offset : offset + len(samples)] += samples

    # Normalize to prevent clipping
    max_val = np.max(np.abs(mix))
    if max_val > 0:
        mix = mix / max_val * 0.9

    # Convert to 16-bit PCM
    mix_int16 = np.int16(mix * MAX_AMPLITUDE)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(output_path), sample_rate, mix_int16)


def notes_to_mp3(
    notes: List[NoteEvent],
    output_path: str | Path,
    sample_rate: int = SAMPLE_RATE,
    bitrate: str = "192k",
) -> None:
    """
    Convert note events to an MP3 file.
    
    First renders to WAV via additive synthesis, then converts to MP3
    using ffmpeg (bundled via imageio-ffmpeg).
    
    Args:
        notes: List of (start_time, pitch, velocity, end_time) tuples
        output_path: Path to save the MP3 file
        sample_rate: Audio sample rate
        bitrate: MP3 bitrate (default "192k")
    """
    if not notes:
        print("Warning: No notes to synthesize")
        return

    # Write a temp WAV first
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = Path(tmp.name)
    try:
        notes_to_wav(notes, tmp_wav, sample_rate=sample_rate)

        # Find ffmpeg from imageio_ffmpeg
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = "ffmpeg"  # fallback to PATH

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            ffmpeg_exe,
            "-y",                    # overwrite
            "-i", str(tmp_wav),      # input WAV
            "-b:a", bitrate,         # audio bitrate
            "-vn",                   # no video
            str(output_path),
        ]
        subprocess.run(cmd, capture_output=True, check=True)
    finally:
        # Clean up temp WAV
        if tmp_wav.exists():
            tmp_wav.unlink()


def notes_to_audio(
    notes: List[NoteEvent],
    output_path: str | Path,
    fmt: str = "mp3",
    **kwargs,
) -> None:
    """
    Convert notes to an audio file (auto-selects WAV or MP3).
    
    Args:
        notes: List of NoteEvents
        output_path: Output file path
        fmt: "wav" or "mp3" (default "mp3")
    """
    if fmt == "mp3":
        notes_to_mp3(notes, output_path, **kwargs)
    else:
        notes_to_wav(notes, output_path, **kwargs)


def midi_file_to_wav(midi_path: str | Path, wav_path: str | Path) -> None:
    """Convert a MIDI file to WAV."""
    notes = midi_file_to_notes(midi_path)
    if notes:
        notes_to_wav(notes, wav_path)


def midi_file_to_mp3(midi_path: str | Path, mp3_path: str | Path) -> None:
    """Convert a MIDI file to MP3."""
    notes = midi_file_to_notes(midi_path)
    if notes:
        notes_to_mp3(notes, mp3_path)


def main():
    """CLI entry point for converting MIDI to audio."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert MIDI to audio (WAV or MP3)")
    parser.add_argument("input", type=str, help="Input MIDI file")
    parser.add_argument("output", type=str, nargs="?", help="Output audio file")
    parser.add_argument("--format", "-f", choices=["wav", "mp3"], default=None,
                       help="Output format (auto-detected from extension)")
    parser.add_argument("--bitrate", "-b", default="192k", help="MP3 bitrate")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        fmt = args.format or "mp3"
        output_path = input_path.with_suffix(f".{fmt}")

    # Determine format
    fmt = args.format or output_path.suffix.lstrip(".").lower()
    if fmt == "mp3":
        midi_file_to_mp3(input_path, output_path)
        print(f"Saved MP3: {output_path}")
    else:
        midi_file_to_wav(input_path, output_path)
        print(f"Saved WAV: {output_path}")


if __name__ == "__main__":
    main()
