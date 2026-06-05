"""
Configuration for the AI Music Generation project.
Adjust these parameters to control training, model architecture, and data processing.
"""
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"          # Downloaded/extracted MIDI files
PROCESSED_DATA_DIR = DATA_DIR / "processed"  # Preprocessed tokenized data
OUTPUT_DIR = PROJECT_ROOT / "output"      # Generated MIDI files
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"  # Model checkpoints

# Create directories
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Dataset ---
# Which MAESTRO dataset version to download
MAESTRO_VERSION = "maestro-v3.0.0"
MAESTRO_URL = f"https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/{MAESTRO_VERSION}-midi.zip"

# --- MIDI Preprocessing ---
# Minimum/maximum MIDI pitch to consider (piano range: 21-108)
MIN_PITCH = 21
MAX_PITCH = 108
N_PITCHES = MAX_PITCH - MIN_PITCH + 1  # 88 keys

# Velocity range (0-127), quantized into bins
N_VELOCITY_BINS = 32

# Time resolution: time shifts in milliseconds
BEAT_RESOLUTION = 4  # ticks per beat (16th notes)
MAX_SHIFT_BEATS = 32  # maximum time shift in beats

# Minimum note length to filter out very short notes (seconds)
MIN_NOTE_LENGTH = 0.05

# --- Tokenization (REMI-style) ---
TOKEN_PAD = 0
TOKEN_BOS = 1  # Beginning of sequence
TOKEN_EOS = 2  # End of sequence
TOKEN_MASK = 3

SPECIAL_TOKENS = 4  # pad, bos, eos, mask

# --- Model Architecture ---
VOCAB_SIZE = None  # Will be computed from the tokenizer
# --- Quick CPU Training Settings (override for speed) ---
D_MODEL = 128      # Embedding dimension (512 for full training)
N_LAYERS = 2       # Number of transformer layers (6 for full training)
N_HEADS = 4        # Number of attention heads (8 for full training)
D_FF = 512         # Feed-forward hidden dimension (2048 for full training)
DROPOUT = 0.1
MAX_SEQUENCE_LEN = 1024  # Maximum sequence length (2048 for full training)

# Relative positional encoding
MAX_RELATIVE_POSITION = 128

# --- Training ---
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 3e-4
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 12
LOG_INTERVAL = 20
SAVE_INTERVAL = 200  # Save checkpoint every N steps

# --- Generation ---
TEMPERATURE = 1.0
TOP_K = 40          # Top-k sampling
TOP_P = 0.9         # Nucleus (top-p) sampling
GENERATE_LENGTH = 512  # Number of tokens to generate
TEMPERATURE_RANGE = (0.8, 1.2)  # Can vary temperature during generation

# --- Device ---
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"
