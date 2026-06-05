"""
AI Music Generator Web UI.

Clean interface with in-browser audio preview, auto-loads model,
and opens automatically in Chrome.
"""
import sys
import webbrowser
from pathlib import Path
from typing import Optional, Tuple, List

import gradio as gr
import numpy as np
import torch

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
from utils.midi_utils import (
    notes_to_midi, save_midi, NoteEvent
)
from utils.midi_to_audio import notes_to_mp3

# ── Globals ───────────────────────────────────────────────────────────
_model: Optional[MusicTransformer] = None
_tokenizer: Optional[REMITokenizer] = None
_model_loaded: bool = False
_vocab_size: int = 0


# ── Helpers ───────────────────────────────────────────────────────────

def _get_default_checkpoint() -> str:
    best = CHECKPOINT_DIR / "best_model.pt"
    if best.exists():
        return str(best)
    step_files = sorted(
        CHECKPOINT_DIR.glob("checkpoint_step_*.pt"),
        key=lambda f: int(f.stem.split("_")[-1]),
    )
    return str(step_files[-1]) if step_files else str(best)


def load_model_and_tokenizer(checkpoint_path: str, tokenizer_path: str) -> Tuple[str, bool]:
    global _model, _tokenizer, _model_loaded, _vocab_size

    ckpt = Path(checkpoint_path)
    tok = Path(tokenizer_path)

    if not ckpt.exists():
        return f"[ERROR] Checkpoint not found: {ckpt}", False
    if not tok.exists():
        return f"[ERROR] Tokenizer not found at {tok}", False

    try:
        _tokenizer = REMITokenizer.load(tok)
        _vocab_size = _tokenizer.vocab_size

        checkpoint = torch.load(ckpt, map_location=DEVICE, weights_only=True)
        cfg = checkpoint.get("config", {})

        model = MusicTransformer(
            vocab_size=_vocab_size,
            d_model=cfg.get("d_model", D_MODEL),
            n_layers=cfg.get("n_layers", N_LAYERS),
            n_heads=cfg.get("n_heads", N_HEADS),
            d_ff=cfg.get("d_ff", D_FF),
            dropout=cfg.get("dropout", DROPOUT),
            max_seq_len=cfg.get("max_seq_len", MAX_SEQUENCE_LEN),
            max_relative_position=cfg.get("max_relative_position", MAX_RELATIVE_POSITION),
            use_rope=False,
        ).to(DEVICE)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        _model = model
        _model_loaded = True

        step = checkpoint.get("step", 0)
        loss = checkpoint.get("loss", 0)
        n_params = model.get_num_params()
        return (
            f"Ready | Vocab: {_vocab_size} | Params: {n_params:,} | Step: {step} | Device: {DEVICE}",
            True,
        )
    except Exception as e:
        _model = None
        _model_loaded = False
        return f"Error: {e}", False


def _generate_single(
    model, tokenizer, prompt_tokens, length, temperature, top_k, top_p,
) -> Tuple[List[int], List[NoteEvent]]:
    with torch.no_grad():
        if prompt_tokens:
            prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=DEVICE)
        else:
            prompt_tensor = torch.tensor([[TOKEN_BOS]], dtype=torch.long, device=DEVICE)

        generated = model.generate(
            prompt=prompt_tensor,
            max_new_tokens=length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=2,
        )

    ids = generated[0].tolist()
    if 2 in ids:
        ids = ids[: ids.index(2) + 1]

    notes = tokenizer.decode_to_notes(ids)
    return ids, notes


def render_piano_roll(notes: List[NoteEvent], filename: str) -> Optional[str]:
    if not notes:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        start_times = np.array([n[0] for n in notes])
        pitches = np.array([n[1] for n in notes])
        end_times = np.array([n[3] for n in notes])

        min_pitch = max(21, pitches.min() - 2)
        max_pitch = min(108, pitches.max() + 2)

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        for note, pitch, start, end in zip(notes, pitches, start_times, end_times):
            duration = max(end - start, 0.05)
            color = plt.cm.viridis((pitch - min_pitch) / max(max_pitch - min_pitch, 1))
            rect = Rectangle(
                (start, pitch - 0.4), duration, 0.8,
                facecolor=color, edgecolor="white", linewidth=0.3, alpha=0.85,
            )
            ax.add_patch(rect)

        ax.set_xlim(start_times.min(), end_times.max() + 0.5)
        ax.set_ylim(min_pitch - 1, max_pitch + 1)
        ax.set_xlabel("Time (seconds)", color="white", fontsize=11)
        ax.set_ylabel("Pitch", color="white", fontsize=11)
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("#444")
        ax.spines["left"].set_color("#444")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.15, color="white")
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        out_path = OUTPUT_DIR / "web_generated" / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return str(out_path)
    except ImportError:
        return None


# ── File serving (Gradio 6.x requires explicit path registration) ──
gr.set_static_paths([str(OUTPUT_DIR)])

# ── Build UI ──────────────────────────────────────────────────────────
MAX_PIECES = 6

CUSTOM_CSS = """
    .gradio-container { max-width: 960px !important; margin: auto; }
    .generate-btn { background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
                    border: none !important; color: white !important;
                    font-weight: 600 !important; font-size: 1.1rem !important;
                    padding: 0.75rem 2rem !important; width: 100% !important; }
    .generate-btn:hover { transform: scale(1.02); box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important; }
    h1 { background: linear-gradient(135deg, #4338ca, #7c3aed);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
         font-weight: 700 !important; }
    .audio-row { background: #f8f9ff; border-radius: 12px; padding: 12px 16px;
                 margin: 6px 0; border-left: 4px solid #6366f1; }
"""

with gr.Blocks(title="AI Music Generator") as demo:

    # ── Header ──────────────────────────────────────────────────────
    gr.Markdown("# AI Music Generator")
    gr.Markdown("Generate original piano music with a transformer model. Click **Generate** to create new pieces.")

    # ── Model status bar (compact) ──────────────────────────────────
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            model_status = gr.Textbox(
                label="Model", value="[PENDING] Not loaded",
                interactive=False, show_label=False,
            )
        with gr.Column(scale=1, min_width=100):
            load_btn = gr.Button("Load Model", variant="secondary", size="sm")

    # ── Generation parameters ──────────────────────────────────────
    with gr.Row():
        temperature = gr.Slider(
            minimum=0.1, maximum=2.0, value=TEMPERATURE, step=0.05,
            label="Temperature", info="Creativity (low=conservative, high=wild)",
        )
        top_k = gr.Slider(
            minimum=0, maximum=100, value=TOP_K, step=1,
            label="Top-K", info="Limit to top K tokens",
        )
        top_p = gr.Slider(
            minimum=0.0, maximum=1.0, value=TOP_P, step=0.05,
            label="Top-P", info="Nucleus sampling threshold",
        )
    with gr.Row():
        length = gr.Slider(
            minimum=64, maximum=2048, value=GENERATE_LENGTH, step=64,
            label="Length", info="More tokens = longer piece",
        )
        num_pieces = gr.Slider(
            minimum=1, maximum=MAX_PIECES, value=4, step=1,
            label="Pieces", info="Number of pieces to generate",
        )

    # ── Generate button ─────────────────────────────────────────────
    generate_btn = gr.Button("Generate Music", variant="primary", elem_classes=["generate-btn"])
    generate_status = gr.Textbox(label="Status", value="Ready", interactive=False)

    # ── Audio previews ──────────────────────────────────────────────
    gr.Markdown("### Preview & Download")
    audio_outputs = []
    for i in range(MAX_PIECES):
        with gr.Row(visible=False, elem_classes=["audio-row"]) as row:
            with gr.Column(scale=1, min_width=70):
                gr.Markdown(f"**Piece {i+1}**")
            with gr.Column(scale=5):
                audio = gr.Audio(
                    label="", type="filepath", show_label=False,
                )
        audio_outputs.append((row, audio))

    # ── Download section ──────────────────────────────────────────
    gr.Markdown("### Download MP3 Files")
    gr.Markdown("Click the files below to download — these work in Windows Media Player, Chrome, and most media players.")
    download_files = gr.File(
        label="Generated MP3 files (click to download)",
        file_count="multiple",
        height=120,
    )

    # ── Piano roll ──────────────────────────────────────────────────
    piano_roll_output = gr.Image(label="Visualization", type="filepath", height=320)

    with gr.Row():
        gr.Markdown(
            '<div style="text-align:center;opacity:0.5;font-size:0.85rem;margin-top:1.5rem">'
            "Built with PyTorch + Music Transformer</div>"
        )

    # ── Event handlers ──────────────────────────────────────────────

    def on_load(checkpoint: str, tokenizer: str) -> str:
        status, ok = load_model_and_tokenizer(checkpoint, tokenizer)
        return status

    load_btn.click(
        fn=on_load,
        inputs=[gr.State(_get_default_checkpoint()), gr.State(str(PROCESSED_DATA_DIR / "tokenizer.json"))],
        outputs=[model_status],
    )

    def on_generate(temp, k, p, gen_len, n_pieces, progress: gr.Progress = gr.Progress()):
        if not _model_loaded:
            ckpt = _get_default_checkpoint()
            tok = str(PROCESSED_DATA_DIR / "tokenizer.json")
            load_status, ok = load_model_and_tokenizer(ckpt, tok)
            if not ok:
                return [load_status, load_status, None] + [gr.update(visible=False)] * (MAX_PIECES * 2) + [None]

        if _model is None or _tokenizer is None:
            return ["Model not loaded", "Not loaded", None] + [gr.update(visible=False)] * (MAX_PIECES * 2) + [None]

        out_dir = OUTPUT_DIR / "web_generated"
        out_dir.mkdir(parents=True, exist_ok=True)
        all_notes = []
        mp3_paths = []

        for i in range(n_pieces):
            progress((0.1 + 0.8 * (i / n_pieces)), desc=f"Generating piece {i+1}/{n_pieces}")
            temp_varied = temp * (0.85 + 0.3 * np.random.random())

            ids, notes = _generate_single(
                _model, _tokenizer, None,
                length=gen_len, temperature=temp_varied,
                top_k=k, top_p=p,
            )
            all_notes.append(notes)

            midi_path = out_dir / f"piece_{i+1}.mid"
            mp3_path = out_dir / f"piece_{i+1}.mp3"

            midi = notes_to_midi(notes, tempo=120.0, program=0)
            save_midi(midi, midi_path)
            try:
                notes_to_mp3(notes, mp3_path)
            except Exception as e:
                print(f"Warning: MP3 conversion failed for piece {i+1}: {e}")
            mp3_paths.append(str(mp3_path) if mp3_path.exists() else None)

        progress(0.95, desc="Rendering preview...")

        plot_path = render_piano_roll(all_notes[-1], "piano_roll.png")

        updates = []
        for i in range(MAX_PIECES):
            if i < n_pieces and mp3_paths[i] is not None:
                updates.append(gr.update(visible=True))
                updates.append(gr.update(value=mp3_paths[i]))
            else:
                updates.append(gr.update(visible=False))
                updates.append(gr.update(value=None))

        total_notes = sum(len(ns) for ns in all_notes)
        # Collect all existing MP3 paths for the file download component
        existing_mp3s = [str(out_dir / f"piece_{i+1}.mp3") for i in range(n_pieces) if (out_dir / f"piece_{i+1}.mp3").exists()]
        mp3_list = existing_mp3s if existing_mp3s else None
        status = f"Done! {n_pieces} piece(s), {total_notes} total notes. Files saved to: {out_dir}"
        return [status, model_status.value] + [mp3_list] + updates + [plot_path]

    audio_inputs = [temperature, top_k, top_p, length, num_pieces]
    audio_output_list = [generate_status, model_status, download_files]
    for row, _ in audio_outputs:
        audio_output_list.append(row)
        audio_output_list.append(_)
    audio_output_list.append(piano_roll_output)

    generate_btn.click(
        fn=on_generate,
        inputs=audio_inputs,
        outputs=audio_output_list,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch AI Music Generator")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    args = parser.parse_args()

    url = f"http://localhost:{args.port}"
    print(f"AI Music Generator — {url}")

    if not args.no_browser:
        webbrowser.open(url)

    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_port=args.port,
        share=False,
        show_error=True,

        allowed_paths=[str(OUTPUT_DIR)],
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
            neutral_hue="gray",
            font=("Inter", "sans-serif"),
        ),
    )
