"""
Training script for the Music Transformer model.

Trains the model on preprocessed token sequences with:
- Learning rate warmup + cosine decay
- Gradient accumulation
- Checkpoint saving
- Loss logging
"""
import sys
import time
import math
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    PROCESSED_DATA_DIR, CHECKPOINT_DIR, OUTPUT_DIR,
    D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT,
    MAX_SEQUENCE_LEN, MAX_RELATIVE_POSITION,
    BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, LEARNING_RATE,
    WARMUP_STEPS, WEIGHT_DECAY, NUM_EPOCHS, LOG_INTERVAL, SAVE_INTERVAL,
    DEVICE
)
from model.tokenizer import REMITokenizer
from model.transformer import MusicTransformer
from model.dataset import MusicTokenDataset


def get_cosine_schedule_with_warmup(
    optimizer, warmup_steps: int, total_steps: int
) -> LambdaLR:
    """Create a learning rate schedule with linear warmup and cosine decay."""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda)


def train():
    """Main training loop."""
    print("=" * 60)
    print("Music Transformer - Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    # --- Load tokenizer ---
    tokenizer_path = PROCESSED_DATA_DIR / "tokenizer.json"
    if tokenizer_path.exists():
        tokenizer = REMITokenizer.load(tokenizer_path)
        vocab_size = tokenizer.vocab_size
        print(f"Loaded tokenizer (vocab size: {vocab_size})")
    else:
        tokenizer = REMITokenizer()
        vocab_size = tokenizer.vocab_size
        print(f"Using default tokenizer (vocab size: {vocab_size})")
    
    # --- Load dataset ---
    data_path = PROCESSED_DATA_DIR / "tokens.npy"
    if not data_path.exists():
        print(f"ERROR: No preprocessed data found at {data_path}")
        print("Please run 'python data/preprocess.py' first.")
        return
    
    print(f"Loading dataset from {data_path}...")
    dataset = MusicTokenDataset(data_path, max_seq_len=MAX_SEQUENCE_LEN)
    print(f"Dataset size: {len(dataset)} chunks")
    
    if len(dataset) == 0:
        print("ERROR: Dataset is empty. Please check your preprocessing.")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # 0 for Windows compatibility
        pin_memory=(DEVICE == "cuda"),
        drop_last=True,
    )
    
    # Update vocab_size based on actual data
    data_vocab_size = dataset.get_vocab_size(tokenizer_path)
    if data_vocab_size > vocab_size:
        print(f"Updating vocab size from {vocab_size} to {data_vocab_size}")
        vocab_size = data_vocab_size
    
    # --- Build model ---
    model = MusicTransformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQUENCE_LEN,
        max_relative_position=MAX_RELATIVE_POSITION,
        use_rope=False,
    ).to(DEVICE)
    
    n_params = model.get_num_params()
    n_trainable = model.get_num_trainable_params()
    print(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")
    
    # --- Optimizer ---
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )
    
    # --- Learning rate scheduler ---
    total_steps = NUM_EPOCHS * len(dataloader) // GRADIENT_ACCUMULATION_STEPS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps=WARMUP_STEPS, total_steps=total_steps
    )
    
    # --- Loss function ---
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    # --- Training loop ---
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Optimizer steps: {total_steps}")
    print()
    
    global_step = 0
    best_loss = float("inf")
    start_time = time.time()
    
    model.train()
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_start = time.time()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            # Forward pass
            logits = model(input_ids)
            loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
            
            # Scale loss for gradient accumulation
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            # Update metrics
            epoch_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            epoch_steps += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                "step": global_step,
            })
            
            # Logging
            if global_step > 0 and global_step % LOG_INTERVAL == 0:
                elapsed = time.time() - start_time
                avg_loss = epoch_loss / epoch_steps
                print(f"\n[Step {global_step}] Loss: {avg_loss:.4f}, "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}, "
                      f"Time: {elapsed:.1f}s")
            
            # Save checkpoint
            if global_step > 0 and global_step % SAVE_INTERVAL == 0:
                checkpoint = {
                    "step": global_step,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": epoch_loss / epoch_steps,
                    "config": {
                        "vocab_size": vocab_size,
                        "d_model": D_MODEL,
                        "n_layers": N_LAYERS,
                        "n_heads": N_HEADS,
                        "d_ff": D_FF,
                        "dropout": DROPOUT,
                        "max_seq_len": MAX_SEQUENCE_LEN,
                        "max_relative_position": MAX_RELATIVE_POSITION,
                    }
                }
                checkpoint_path = CHECKPOINT_DIR / f"checkpoint_step_{global_step}.pt"
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
        
        # End of epoch
        epoch_loss_avg = epoch_loss / epoch_steps
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1} completed in {epoch_time:.1f}s")
        print(f"  Average loss: {epoch_loss_avg:.4f}")
        
        # Save epoch checkpoint
        if epoch_loss_avg < best_loss:
            best_loss = epoch_loss_avg
            best_path = CHECKPOINT_DIR / "best_model.pt"
            torch.save({
                "step": global_step,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": best_loss,
            }, best_path)
            print(f"  New best model! Saved to {best_path}")
    
    # --- Final save ---
    final_path = CHECKPOINT_DIR / "final_model.pt"
    torch.save({
        "step": global_step,
        "epoch": NUM_EPOCHS,
        "model_state_dict": model.state_dict(),
        "loss": epoch_loss / epoch_steps if epoch_steps > 0 else 0,
        "config": {
            "vocab_size": vocab_size,
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "d_ff": D_FF,
            "dropout": DROPOUT,
            "max_seq_len": MAX_SEQUENCE_LEN,
        }
    }, final_path)
    
    total_time = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Training complete! Total time: {total_time:.1f}s")
    print(f"Final model saved to: {final_path}")
    print(f"Best model saved to: {CHECKPOINT_DIR / 'best_model.pt'}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    train()
