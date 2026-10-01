import os
import sys
import argparse
import datetime
import torch

def format_bytes(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def list_checkpoints(checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        print(f"No checkpoint directory found at '{checkpoint_dir}'.")
        return []
    
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith(('.pth', '.pt'))]
    if not files:
        print(f"No checkpoint files (.pth/.pt) found in '{checkpoint_dir}'.")
        return []

    print(f"\n{'='*70}")
    print(f" Available Checkpoints in '{checkpoint_dir}/'")
    print(f"{'='*70}")
    print(f" {'File Name':<35} | {'Size':<12} | {'Last Modified':<20}")
    print(f"{'-'*70}")
    
    for f in sorted(files):
        full_path = os.path.join(checkpoint_dir, f)
        stat = os.stat(full_path)
        mtime = datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        size_str = format_bytes(stat.st_size)
        print(f" {f:<35} | {size_str:<12} | {mtime:<20}")
    print(f"{'='*70}\n")
    return files

def inspect_checkpoint(checkpoint_path, verbose=False):
    if not os.path.exists(checkpoint_path):
        print(f"Error: File not found at '{checkpoint_path}'")
        return

    print(f"\n{'='*70}")
    print(f" Checkpoint Details: {checkpoint_path}")
    print(f"{'='*70}")

    file_size = format_bytes(os.path.getsize(checkpoint_path))
    print(f" File Size        : {file_size}")

    # Load state dict
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    total_params = sum(p.numel() for p in state_dict.values())
    total_mem_mb = sum(p.numel() * p.element_size() for p in state_dict.values()) / (1024 * 1024)

    print(f" Total Parameters : {total_params:,} ({total_params/1e6:.2f} M)")
    print(f" Weights Memory   : {total_mem_mb:.2f} MB")
    print(f" Total Tensors    : {len(state_dict)}")
    print(f"{'-'*70}")

    # Group components
    components = {}
    for key, tensor in state_dict.items():
        prefix = key.split('.')[0]
        if prefix not in components:
            components[prefix] = []
        components[prefix].append((key, tensor))

    print(" Architecture Breakdown:")
    for comp, layers in components.items():
        comp_params = sum(t.numel() for _, t in layers)
        print(f"   - {comp:<20} : {len(layers):>3} tensors | {comp_params:>10,} params ({comp_params/total_params*100:5.1f}%)")

    print(f"{'-'*70}")
    if verbose:
        print(" Detailed Layer Shapes:")
        for name, tensor in state_dict.items():
            shape_str = str(list(tensor.shape))
            print(f"   {name:<50} : shape {shape_str:<20} ({tensor.dtype})")
    else:
        print(" First 8 Layers Sample (use --verbose to see all):")
        for i, (name, tensor) in enumerate(list(state_dict.items())[:8]):
            shape_str = str(list(tensor.shape))
            print(f"   {name:<50} : shape {shape_str:<20}")
        print(f"   ... ({len(state_dict) - 8} more layers)")
    print(f"{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(description="Inspect PyTorch Checkpoint Files")
    parser.add_argument("path", nargs="?", default="checkpoints/foundation_encoder_best.pth",
                        help="Path to checkpoint file (.pth). Default: checkpoints/foundation_encoder_best.pth")
    parser.add_argument("--list", action="store_true", help="List all available checkpoint files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all layer names and shapes")
    args = parser.parse_args()

    list_checkpoints()

    if not args.list and os.path.exists(args.path):
        inspect_checkpoint(args.path, verbose=args.verbose)

if __name__ == "__main__":
    main()
