"""
Download MAESTRO dataset and/or provide instructions for other datasets.

The MAESTRO dataset contains over 200 hours of piano performances with
alignments between MIDI and audio recordings.
"""
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DATA_DIR, MAESTRO_URL, MAESTRO_VERSION


def download_maestro(url: str = MAESTRO_URL, 
                     dest_dir: str | Path = RAW_DATA_DIR,
                     version: str = MAESTRO_VERSION) -> Path:
    """
    Download the MAESTRO MIDI dataset.
    
    Note: This downloads from Google Cloud Storage which may require
    authentication or have rate limits. If the automatic download fails,
    follow the manual instructions below.
    
    Args:
        url: Download URL
        dest_dir: Destination directory
        version: Dataset version string
    
    Returns:
        Path to extracted dataset directory
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = dest_dir / f"{version}-midi.zip"
    extract_dir = dest_dir / version
    
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"Dataset already exists at {extract_dir}")
        return extract_dir
    
    if not zip_path.exists():
        print(f"Downloading MAESTRO dataset from {url}...")
        print(f"File will be saved to {zip_path}")
        print(f"Size: ~1.5GB")
        
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024 * 1024  # 1MB chunks
            
            with open(zip_path, "wb") as f:
                with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print("Download complete!")
        except Exception as e:
            print(f"Download failed: {e}")
            print("\nManual download instructions:")
            print(f"1. Go to: https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/")
            print(f"2. Download {version}-midi.zip")
            print(f"3. Place it in: {dest_dir}")
            print(f"4. Re-run this script to extract it.")
            return extract_dir
    
    # Extract the zip file
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print(f"Extracted to {extract_dir}")
    
    return extract_dir


def list_midi_files(directory: str | Path) -> list[Path]:
    """Recursively list all MIDI files in a directory."""
    directory = Path(directory)
    midi_files = []
    
    for ext in ["*.mid", "*.midi", "*.MID", "*.MIDI"]:
        midi_files.extend(directory.rglob(ext))
    
    return sorted(midi_files)


def get_dataset_stats(midi_files: list[Path]) -> dict:
    """Get basic statistics about a MIDI dataset."""
    import pretty_midi
    
    stats = {
        "total_files": len(midi_files),
        "total_duration": 0.0,
        "total_notes": 0,
        "instruments_per_file": [],
    }
    
    for f in tqdm(midi_files, desc="Analyzing dataset"):
        try:
            midi = pretty_midi.PrettyMIDI(str(f))
            n_notes = sum(len(inst.notes) for inst in midi.instruments if not inst.is_drum)
            duration = max(
                (max(n.end for n in inst.notes) if inst.notes else 0)
                for inst in midi.instruments
            ) if midi.instruments else 0
            
            stats["total_notes"] += n_notes
            stats["total_duration"] += duration
            stats["instruments_per_file"].append(
                sum(1 for i in midi.instruments if not i.is_drum)
            )
        except Exception:
            pass
    
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download MIDI dataset")
    parser.add_argument("--dataset", choices=["maestro", "lakh"], default="maestro",
                       help="Dataset to download")
    
    args = parser.parse_args()
    
    if args.dataset == "maestro":
        data_dir = download_maestro()
        midi_files = list_midi_files(data_dir)
        print(f"\nFound {len(midi_files)} MIDI files")
        
        if midi_files:
            stats = get_dataset_stats(midi_files[:10])  # Sample first 10
            print(f"Sample stats (first 10 files):")
            print(f"  Total notes: {stats['total_notes']}")
            print(f"  Total duration: {stats['total_duration']:.1f}s")
    else:
        print("Lakh MIDI dataset: https://colinraffel.com/projects/lmd/")
        print("Download from: https://huggingface.co/datasets/sodascience/lakh_midi")
