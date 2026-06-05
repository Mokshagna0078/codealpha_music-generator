"""
Music generation script.

Loads a trained Music Transformer model and generates new music sequences,
saving them as MIDI files.
"""
import sys
from pathlib import Path
from typing import Optional, List
import torch
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CHECKPOINT_DIR, OUTPUT_DIR, PROCESSED_DATA_DIR,
    D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT,
    MAX_SEQUENCE_LEN, MAX_RELATIVE_POSITION,
    TEMPERATURE, TOP_K, TOP_P, GENERATE_LENGTH,
    TOKEN_BOS, DEVICE
)
from model.tokenizer import REMITokenizer
from model.transformer import MusicTransformer
from utils.midi_utils import notes_to_midi, save_midi, NoteEvent
from utils.midi_to_audio import notes_to_mp3


def load_model(
    checkpoint_path: str | Path,
    vocab_size: int,
    device: str = DEVICE,
) -> MusicTransformer:
    """
    Load a trained model from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        vocab_size: Vocabulary size
        device: Device to load the model on
    
    Returns:
        Loaded MusicTransformer model in eval mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Get config from checkpoint
    config = checkpoint.get("config", {})
    
    model = MusicTransformer(
        vocab_size=vocab_size,
        d_model=config.get("d_model", D_MODEL),
        n_layers=config.get("n_layers", N_LAYERS),
        n_heads=config.get("n_heads", N_HEADS),
        d_ff=config.get("d_ff", D_FF),
        dropout=config.get("dropout", DROPOUT),
        max_seq_len=config.get("max_seq_len", MAX_SEQUENCE_LEN),
        max_relative_position=config.get("max_relative_position", MAX_RELATIVE_POSITION),
        use_rope=False,
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    step = checkpoint.get("step", 0)
    loss = checkpoint.get("loss", 0)
    print(f"Loaded checkpoint (step {step}, loss: {loss:.4f})")
    
    return model


def generate_music(
    model: MusicTransformer,
    tokenizer: REMITokenizer,
    prompt: Optional[List[int]] = None,
    length: int = GENERATE_LENGTH,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
    top_p: float = TOP_P,
    eos_token_id: int = 2,
    device: str = DEVICE,
) -> List[int]:
    """
    Generate a music sequence.
    
    Args:
        model: Trained MusicTransformer
        tokenizer: REMITokenizer for decoding
        prompt: Optional initial token sequence (starts with BOS if None)
        length: Number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Nucleus sampling threshold
        eos_token_id: End-of-sequence token
        device: Device to run on
    
    Returns:
        Generated token IDs
    """
    if prompt is None:
        prompt = [TOKEN_BOS]
    
    prompt_tensor = torch.tensor([prompt], dtype=torch.long, device=device)
    
    with torch.no_grad():
        generated = model.generate(
            prompt=prompt_tensor,
            max_new_tokens=length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )
    
    # Convert back to list
    generated_ids = generated[0].tolist()
    
    # Trim to EOS if present
    if eos_token_id in generated_ids:
        eos_idx = generated_ids.index(eos_token_id)
        generated_ids = generated_ids[:eos_idx + 1]
    
    return generated_ids


def save_generated_midi(
    token_ids: List[int],
    tokenizer: REMITokenizer,
    output_path: str | Path,
    beat_duration: float = 0.5,  # 120 BPM
    tempo: float = 120.0,
    program: int = 0,  # Acoustic Grand Piano
):
    """
    Convert generated tokens to MIDI and MP3 files.
    
    MP3 files are compact audio that plays in any media player.
    
    Args:
        token_ids: Generated token IDs
        tokenizer: REMITokenizer for decoding
        output_path: Path to save the MIDI file (MP3 saved alongside)
        beat_duration: Duration of one beat in seconds
        tempo: BPM
        program: MIDI program number
    """
    # Decode tokens to notes
    notes = tokenizer.decode_to_notes(token_ids, beat_duration=beat_duration)
    
    if not notes:
        print("Warning: No notes decoded from generated tokens")
        return
    
    print(f"Generated {len(notes)} notes")
    
    # Convert to MIDI
    midi = notes_to_midi(notes, tempo=tempo, program=program)
    
    # Save MIDI
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_midi(midi, output_path)
    print(f"Saved MIDI: {output_path}")
    
    # Save MP3 alongside (same name, .mp3 extension) — much smaller than WAV
    mp3_path = output_path.with_suffix('.mp3')
    try:
        notes_to_mp3(notes, mp3_path)
        print(f"Saved MP3:  {mp3_path} (plays in any media player)")
    except Exception as e:
        print(f"Warning: Could not save MP3: {e}")


def generate_multiple(
    model: MusicTransformer,
    tokenizer: REMITokenizer,
    num_generations: int = 4,
    length: int = GENERATE_LENGTH,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
    top_p: float = TOP_P,
    output_dir: str | Path = OUTPUT_DIR,
    base_name: str = "generated",
    device: str = DEVICE,
) -> List[Path]:
    """
    Generate multiple music pieces and save them (MIDI + MP3 each).
    
    Returns:
        List of paths to generated MIDI files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    for i in range(num_generations):
        # Vary temperature slightly for diversity
        temp = temperature * (0.8 + 0.4 * np.random.random())
        
        print(f"\nGenerating piece {i+1}/{num_generations} (temp={temp:.2f})...")
        
        # Generate
        token_ids = generate_music(
            model, tokenizer,
            length=length,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            device=device,
        )
        
        # Save (MIDI + MP3)
        output_path = output_dir / f"{base_name}_{i+1}.mid"
        save_generated_midi(token_ids, tokenizer, output_path)
        saved_paths.append(output_path)
    
    return saved_paths


def main():
    """Main entry point for generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate music with trained model")
    parser.add_argument("--checkpoint", type=str, 
                       default=str(CHECKPOINT_DIR / "best_model.pt"),
                       help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str,
                       default=str(PROCESSED_DATA_DIR / "tokenizer.json"),
                       help="Path to tokenizer JSON")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                       help="Output directory for MIDI files")
    parser.add_argument("--num-pieces", type=int, default=4,
                       help="Number of pieces to generate")
    parser.add_argument("--length", type=int, default=GENERATE_LENGTH,
                       help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                       help="Sampling temperature (0.1-2.0)")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                       help="Top-K filtering (0 to disable)")
    parser.add_argument("--top-p", type=float, default=TOP_P,
                       help="Top-p nucleus sampling (1.0 to disable)")
    parser.add_argument("--prompt", type=str, default=None,
                       help="MIDI file to use as prompt (optional)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Music Generation with AI")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    # Load tokenizer
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        print(f"Tokenizer not found at {tokenizer_path}, using default")
        tokenizer = REMITokenizer()
    else:
        tokenizer = REMITokenizer.load(tokenizer_path)
    
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    
    # Load model — auto-detect latest checkpoint if default not found
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        # Try to find the latest checkpoint_step_XXX.pt file
        ckpt_dir = CHECKPOINT_DIR
        step_files = sorted(ckpt_dir.glob("checkpoint_step_*.pt"),
                        key=lambda f: int(f.stem.split("_")[-1]))
        if step_files:
            checkpoint_path = step_files[-1]
            print(f"Default checkpoint not found. Using latest: {checkpoint_path.name}")
        else:
            print(f"ERROR: No checkpoint found in {CHECKPOINT_DIR}")
            print("Please train a model first: python train.py")
            return
    
    model = load_model(checkpoint_path, vocab_size, device=DEVICE)
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Load prompt if provided
    prompt_tokens = None
    if args.prompt:
        from utils.midi_utils import load_midi, extract_notes
        print(f"Loading prompt from: {args.prompt}")
        midi = load_midi(args.prompt)
        if midi is not None:
            notes = extract_notes(midi)
            if notes:
                prompt_tokens = tokenizer.encode_notes(notes, add_special=False)
                # Prepend BOS token for the model
                prompt_tokens = [TOKEN_BOS] + prompt_tokens
                print(f"Prompt: {len(prompt_tokens)} tokens (with BOS)")
    
    # Generate
    if prompt_tokens:
        # Single generation with prompt
        print(f"\nGenerating with prompt ({len(prompt_tokens)} tokens)...")
        token_ids = generate_music(
            model, tokenizer,
            prompt=prompt_tokens,
            length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=DEVICE,
        )
        output_path = Path(args.output_dir) / "generated_with_prompt.mid"
        save_generated_midi(token_ids, tokenizer, output_path)
        mp3_path = output_path.with_suffix('.mp3')
        print(f"\nDone! Saved:")
        print(f"  MIDI: {output_path}")
        print(f"  MP3:  {mp3_path} (open in any media player)")
    else:
        # Generate multiple pieces
        saved = generate_multiple(
            model, tokenizer,
            num_generations=args.num_pieces,
            length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            output_dir=args.output_dir,
            device=DEVICE,
        )
        print(f"\nDone! Generated {len(saved)} pieces:")
        for path in saved:
            mp3_path = path.with_suffix('.mp3')
            print(f"  - {path} (MIDI)")
            print(f"  - {mp3_path} (MP3 - double-click to play)")


if __name__ == "__main__":
    main()
