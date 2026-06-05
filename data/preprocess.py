"""
Preprocess MIDI files into token sequences for training.

Pipeline:
1. Load MIDI files
2. Extract note events (start, pitch, velocity, end)
3. Quantize timings to a rhythmic grid
4. Encode to REMI token sequences
5. Save as numpy arrays for training
"""
import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
from tqdm import tqdm
import json
import pretty_midi

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MIN_PITCH, MAX_PITCH, MIN_NOTE_LENGTH,
    BEAT_RESOLUTION
)
from utils.midi_utils import (
    load_midi, extract_notes, quantize_notes, get_total_duration,
    merge_instruments_to_piano
)
from model.tokenizer import REMITokenizer, NoteEvent


def preprocess_midi_file(
    midi_path: str | Path,
    tokenizer: REMITokenizer,
    merge_to_piano: bool = True,
    min_pitch: int = MIN_PITCH,
    max_pitch: int = MAX_PITCH,
    min_note_length: float = MIN_NOTE_LENGTH,
    beat_duration: float = 0.5,
    max_duration_seconds: float = 600.0,
) -> Optional[List[int]]:
    """
    Preprocess a single MIDI file into token IDs.
    
    Args:
        midi_path: Path to the MIDI file
        tokenizer: REMITokenizer instance
        merge_to_piano: Whether to merge all instruments to piano
        min_pitch: Minimum MIDI pitch to keep
        max_pitch: Maximum MIDI pitch to keep
        min_note_length: Minimum note duration in seconds
        beat_duration: Duration of one beat in seconds (0.5s = 120 BPM)
        max_duration_seconds: Skip files longer than this
    
    Returns:
        List of token IDs, or None if preprocessing fails
    """
    try:
        midi = load_midi(midi_path)
        if midi is None:
            return None
        
        # Skip very long files
        duration = get_total_duration(midi)
        if duration > max_duration_seconds:
            return None
        
        # Skip files that are too short
        if duration < 5.0:
            return None
        
        # Optionally merge to single piano track
        if merge_to_piano and len(midi.instruments) > 1:
            midi = merge_instruments_to_piano(midi)
        
        # Extract notes
        notes = extract_notes(midi, min_pitch=min_pitch, max_pitch=max_pitch, 
                              min_length=min_note_length)
        
        if len(notes) < 10:
            return None
        
        # Quantize timings
        notes = quantize_notes(notes, ticks_per_beat=BEAT_RESOLUTION, bpm=120.0)
        
        # Encode to tokens
        token_ids = tokenizer.encode_notes(notes, beat_duration=beat_duration)
        
        return token_ids
    
    except Exception as e:
        return None


def preprocess_dataset(
    input_dir: str | Path = RAW_DATA_DIR,
    output_dir: str | Path = PROCESSED_DATA_DIR,
    merge_to_piano: bool = True,
    max_files: Optional[int] = None,
) -> Path:
    """
    Preprocess all MIDI files in a directory.
    
    Args:
        input_dir: Directory containing MIDI files (recursive search)
        output_dir: Directory to save preprocessed data
        merge_to_piano: Whether to merge tracks to piano
        max_files: Maximum number of files to process (None = all)
    
    Returns:
        Path to the saved token data file
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all MIDI files
    midi_files = []
    for ext in ["*.mid", "*.midi", "*.MID", "*.MIDI"]:
        midi_files.extend(input_dir.rglob(ext))
    midi_files = sorted(midi_files)
    
    if not midi_files:
        print(f"No MIDI files found in {input_dir}")
        print("Please run 'python data/download.py' first or place MIDI files in the data/raw directory.")
        return output_dir / "tokens.npy"
    
    if max_files:
        midi_files = midi_files[:max_files]
    
    print(f"Found {len(midi_files)} MIDI files")
    
    # Initialize tokenizer
    tokenizer = REMITokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Process each file
    all_tokens = []
    skipped = 0
    
    for midi_path in tqdm(midi_files, desc="Preprocessing"):
        tokens = preprocess_midi_file(midi_path, tokenizer, merge_to_piano)
        if tokens is not None:
            all_tokens.append(tokens)
        else:
            skipped += 1
    
    print(f"Processed {len(all_tokens)} files, skipped {skipped} files")
    
    # Save tokenized data
    output_path = output_dir / "tokens.npy"
    np.save(output_path, np.array(all_tokens, dtype=object), allow_pickle=True)
    print(f"Saved token sequences to {output_path}")
    
    # Save tokenizer
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    print(f"Saved tokenizer to {tokenizer_path}")
    
    # Save metadata
    metadata = {
        "num_files": len(all_tokens),
        "num_files_skipped": skipped,
        "total_sequences": len(all_tokens),
        "vocab_size": tokenizer.vocab_size,
        "min_pitch": MIN_PITCH,
        "max_pitch": MAX_PITCH,
        "beat_resolution": BEAT_RESOLUTION,
        "max_sequence_lengths": {
            "min": min(len(t) for t in all_tokens) if all_tokens else 0,
            "max": max(len(t) for t in all_tokens) if all_tokens else 0,
            "mean": float(np.mean([len(t) for t in all_tokens])) if all_tokens else 0,
        }
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess MIDI dataset")
    parser.add_argument("--input-dir", type=str, default=str(RAW_DATA_DIR),
                       help="Directory containing MIDI files")
    parser.add_argument("--output-dir", type=str, default=str(PROCESSED_DATA_DIR),
                       help="Directory to save preprocessed data")
    parser.add_argument("--max-files", type=int, default=None,
                       help="Maximum number of files to process")
    parser.add_argument("--no-merge", action="store_true",
                       help="Don't merge instruments to piano")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    output_path = preprocess_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        merge_to_piano=not args.no_merge,
        max_files=args.max_files,
    )
    
    print(f"\nPreprocessing complete! Data saved to: {output_path}")
    print(f"Run 'python train.py' to start training.")
