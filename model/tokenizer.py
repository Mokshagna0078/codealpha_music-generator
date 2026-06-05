"""
REMI (REvamped MIDI) tokenizer for music generation.

Converts MIDI note events into a sequence of discrete tokens:
- Pitch tokens (e.g., PITCH_60)
- Velocity tokens (quantized into bins)
- Duration tokens (quantized time shifts)
- Time-Shift tokens (time elapsed since last event)

Based on the REMI tokenization scheme from "Pop Music Transformer"
and related works.
"""
from typing import List, Tuple, Dict, Optional
import numpy as np
from pathlib import Path
import json

from config import (
    MIN_PITCH, MAX_PITCH, N_PITCHES,
    N_VELOCITY_BINS, BEAT_RESOLUTION, MAX_SHIFT_BEATS,
    SPECIAL_TOKENS, TOKEN_PAD, TOKEN_BOS, TOKEN_EOS
)

# Note event type
NoteEvent = Tuple[float, int, int, float]  # (start_time, pitch, velocity, end_time)


class REMITokenizer:
    """
    REMI-style tokenizer for music sequences.
    
    Token types:
    - Pitch: PITCH_21 through PITCH_108 (88 tokens)
    - Velocity: VEL_0 through VEL_31 (32 bins)
    - Duration: DUR_0 through DUR_* (quantized durations)  
    - Time Shift: SHIFT_0 through SHIFT_* (time between events)
    - Special: PAD, BOS, EOS
    """
    
    def __init__(self):
        self._build_vocab()
    
    def _build_vocab(self):
        """Build the token vocabulary."""
        self.token_to_id = {}
        self.id_to_token = {}
        
        # Special tokens
        self._add_special("PAD", TOKEN_PAD)
        self._add_special("BOS", TOKEN_BOS)
        self._add_special("EOS", TOKEN_EOS)
        
        # Pitch tokens (e.g., PITCH_60)
        for pitch in range(MIN_PITCH, MAX_PITCH + 1):
            self._add_token(f"PITCH_{pitch}")
        
        # Velocity tokens (quantized into bins)
        # MIDI velocity: 0-127, quantized into N_VELOCITY_BINS
        for i in range(N_VELOCITY_BINS):
            self._add_token(f"VEL_{i}")
        
        # Duration tokens (in ticks)
        # Durations up to MAX_SHIFT_BEATS * BEAT_RESOLUTION ticks
        max_duration_ticks = MAX_SHIFT_BEATS * BEAT_RESOLUTION
        for i in range(max_duration_ticks + 1):
            self._add_token(f"DUR_{i}")
        
        # Time-shift tokens (in ticks, same range as durations)
        for i in range(max_duration_ticks + 1):
            self._add_token(f"SHIFT_{i}")
        
        self.vocab_size = len(self.token_to_id)
    
    def _add_token(self, token: str) -> int:
        """Add a token to the vocabulary and return its ID."""
        idx = len(self.token_to_id)
        self.token_to_id[token] = idx
        self.id_to_token[idx] = token
        return idx
    
    def _add_special(self, token: str, idx: int):
        """Add a special token at a specific index."""
        self.token_to_id[token] = idx
        self.id_to_token[idx] = token
    
    def _quantize_velocity(self, velocity: int) -> int:
        """Quantize MIDI velocity (0-127) into a bin (0 to N_VELOCITY_BINS-1)."""
        bin_size = 128 // N_VELOCITY_BINS
        return min(velocity // bin_size, N_VELOCITY_BINS - 1)
    
    def _dequantize_velocity(self, bin_idx: int) -> int:
        """Convert velocity bin back to approximate MIDI velocity."""
        bin_size = 128 // N_VELOCITY_BINS
        return bin_idx * bin_size + bin_size // 2
    
    def _time_to_ticks(self, time_seconds: float, beat_duration: float = 0.5) -> int:
        """
        Convert time in seconds to ticks.
        Default beat_duration = 0.5s corresponds to 120 BPM.
        """
        ticks_per_second = BEAT_RESOLUTION / beat_duration
        return int(round(time_seconds * ticks_per_second))
    
    def _ticks_to_time(self, ticks: int, beat_duration: float = 0.5) -> float:
        """Convert ticks back to seconds."""
        seconds_per_tick = beat_duration / BEAT_RESOLUTION
        return ticks * seconds_per_tick
    
    def encode_notes(self, notes: List[NoteEvent], 
                     beat_duration: float = 0.5,
                     add_special: bool = True) -> List[int]:
        """
        Encode a list of note events into token IDs using REMI format.
        
        Format per note: [PITCH_x, VEL_y, DUR_z]
        Time shifts between notes: [SHIFT_t]
        
        Args:
            notes: List of (start, pitch, velocity, end) tuples, sorted by start time
            beat_duration: Duration of one beat in seconds
            add_special: Whether to add BOS and EOS tokens
        
        Returns:
            List of token IDs
        """
        tokens = []
        
        if add_special:
            tokens.append(TOKEN_BOS)
        
        prev_start = 0.0
        
        for i, (start, pitch, velocity, end) in enumerate(notes):
            # Compute time shift from previous note start
            time_shift = start - prev_start
            shift_ticks = self._time_to_ticks(max(0, time_shift), beat_duration)
            max_ticks = MAX_SHIFT_BEATS * BEAT_RESOLUTION
            shift_ticks = min(shift_ticks, max_ticks)
            
            if shift_ticks > 0:
                tokens.append(self.token_to_id[f"SHIFT_{shift_ticks}"])
            
            # Pitch
            tokens.append(self.token_to_id[f"PITCH_{pitch}"])
            
            # Velocity (quantized)
            vel_bin = self._quantize_velocity(velocity)
            tokens.append(self.token_to_id[f"VEL_{vel_bin}"])
            
            # Duration
            duration = max(0, end - start)
            dur_ticks = self._time_to_ticks(duration, beat_duration)
            max_ticks = MAX_SHIFT_BEATS * BEAT_RESOLUTION
            dur_ticks = min(max(1, dur_ticks), max_ticks)
            tokens.append(self.token_to_id[f"DUR_{dur_ticks}"])
            
            prev_start = start
        
        if add_special:
            tokens.append(TOKEN_EOS)
        
        return tokens
    
    def decode_to_notes(self, token_ids: List[int],
                        beat_duration: float = 0.5) -> List[NoteEvent]:
        """
        Decode a sequence of token IDs back to note events.
        
        Args:
            token_ids: List of token IDs
            beat_duration: Duration of one beat in seconds
        
        Returns:
            List of (start, pitch, velocity, end) tuples
        """
        notes = []
        current_time = 0.0
        i = 0
        
        while i < len(token_ids):
            token_id = token_ids[i]
            token_str = self.id_to_token.get(token_id, "UNKNOWN")
            
            if token_str == "BOS":
                i += 1
                continue
            elif token_str == "EOS":
                break
            elif token_str.startswith("SHIFT_"):
                ticks = int(token_str.split("_")[1])
                current_time += self._ticks_to_time(ticks, beat_duration)
                i += 1
            elif token_str.startswith("PITCH_"):
                pitch = int(token_str.split("_")[1])
                
                # Next token should be velocity
                vel = 64  # default velocity
                if i + 1 < len(token_ids):
                    next_token = self.id_to_token.get(token_ids[i + 1], "")
                    if next_token.startswith("VEL_"):
                        vel_bin = int(next_token.split("_")[1])
                        vel = self._dequantize_velocity(vel_bin)
                
                # Next should be duration
                dur = 0.25  # default duration (quarter note)
                if i + 2 < len(token_ids):
                    next_token2 = self.id_to_token.get(token_ids[i + 2], "")
                    if next_token2.startswith("DUR_"):
                        ticks = int(next_token2.split("_")[1])
                        dur = self._ticks_to_time(ticks, beat_duration)
                        if dur <= 0:
                            dur = self._ticks_to_time(1, beat_duration)
                
                notes.append((current_time, pitch, vel, current_time + dur))
                i += 3  # consume PITCH, VEL, DUR
            else:
                i += 1
        
        return notes
    
    def encode_to_string(self, token_ids: List[int]) -> str:
        """Convert token IDs to human-readable string."""
        return " ".join(self.id_to_token.get(tid, "?") for tid in token_ids)
    
    def save(self, path: str | Path):
        """Save the tokenizer vocabulary to a JSON file."""
        # Convert id_to_token keys to strings for JSON serialization
        id_to_token_str_keys = {str(k): v for k, v in self.id_to_token.items()}
        data = {
            "token_to_id": self.token_to_id,
            "id_to_token": id_to_token_str_keys,
            "vocab_size": self.vocab_size
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str | Path) -> "REMITokenizer":
        """Load a tokenizer from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        tokenizer = cls.__new__(cls)
        tokenizer.token_to_id = data["token_to_id"]
        # id_to_token keys are stored as strings in JSON, convert back to ints
        id_to_token_raw = data.get("id_to_token", {})
        if id_to_token_raw:
            tokenizer.id_to_token = {int(k): v for k, v in id_to_token_raw.items()}
        else:
            # Legacy fallback: reconstruct from token_to_id
            tokenizer.id_to_token = {v: k for k, v in tokenizer.token_to_id.items()}
        tokenizer.vocab_size = data["vocab_size"]
        return tokenizer
    
    @property
    def n_pitch_tokens(self) -> int:
        """Number of pitch tokens."""
        return N_PITCHES
    
    @property
    def n_velocity_tokens(self) -> int:
        """Number of velocity tokens."""
        return N_VELOCITY_BINS
    
    @property
    def n_shift_tokens(self) -> int:
        """Number of time-shift tokens."""
        return MAX_SHIFT_BEATS * BEAT_RESOLUTION + 1
    
    @property
    def n_duration_tokens(self) -> int:
        """Number of duration tokens."""
        return MAX_SHIFT_BEATS * BEAT_RESOLUTION + 1
