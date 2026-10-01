import os
import sys
import json
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, Subset

# Add the 'src' directory to the path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dummy_dataset import DummyECGDataset
from data.masking import apply_block_masking
from data.ptbxl_dataset import PTBXLECGDataset
from data.chapman_dataset import ChapmanECGDataset
from data.mimic_iv_ecg_dataset import MIMICIVECGDataset
from models.foundation_encoder import FoundationEncoder

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained Phase 1 Foundation Encoder on Test ECGs")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/foundation_encoder_phase1.pth",
                        help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--dataset", type=str, default="all3",
                        choices=["all3", "both", "ptbxl", "chapman", "mimic", "dummy"],
                        help="Test dataset: 'all3' (PTB-XL + Chapman + MIMIC-IV-ECG test splits), "
                             "'both' (PTB-XL + Chapman), 'ptbxl', 'chapman', 'mimic', or 'dummy'")
    parser.add_argument("--ptbxl_dir", type=str, default="data/ptbxl", help="Path to PTB-XL dataset directory")
    parser.add_argument("--chapman_dir", type=str, default="data/Chapman", help="Path to Chapman dataset directory")
    parser.add_argument("--mimic_dir", type=str, default="data/MIMIC-IV-ECG", help="Path to MIMIC-IV-ECG dataset directory")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit test samples (for fast evaluation)")
    parser.add_argument("--mask_ratio", type=float, default=0.6, help="Block masking ratio (default: 0.6 per Sec 4.2)")
    parser.add_argument("--patch_size", type=int, default=50, help="Patch token size (default: 50)")
    parser.add_argument("--num_layers", type=int, default=8, help="Transformer encoder depth (default: 8)")
    parser.add_argument("--embed_dim", type=int, default=256, help="Transformer embedding dimension (default: 256)")
    parser.add_argument("--num_heads", type=int, default=8, help="Attention heads (default: 8)")
    parser.add_argument("--save_plot", type=str, default="reports/reconstruction_sample_{idx}.png",
                        help="File path pattern to save waveform plots (use {idx} for sample index)")
    parser.add_argument("--num_plots", type=int, default=5,
                        help="Number of different ECG reconstruction plots to generate (default: 5)")
    parser.add_argument("--output_json", type=str, default="reports/eval_metrics.json",
                        help="File path to save numerical evaluation metrics in JSON format")
    return parser.parse_args()

def evaluate_phase1(args=None):
    if args is None:
        args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"============================================================")
    print(f" Phase 1 Foundation Encoder: Masked Reconstruction Evaluation ")
    print(f"============================================================")
    print(f"Using device       : {device}")
    print(f"Model Checkpoint   : {args.checkpoint}")
    print(f"Mask Ratio         : {args.mask_ratio * 100:.0f}% contiguous block masking")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at '{args.checkpoint}'. Train the model first via train_phase1.py.")

    # 1. Load Test Dataset (held-out splits never seen in training)
    datasets = []
    if args.dataset in ["all3", "both", "ptbxl"]:
        if os.path.exists(os.path.join(args.ptbxl_dir, "ptbxl_database.csv")):
            print(f"Loading PTB-XL held-out test split (fold 10 benchmark)...")
            ptbxl_test = PTBXLECGDataset(data_dir=args.ptbxl_dir, sampling_rate=500, split="test")
            print(f"  -> PTB-XL test records: {len(ptbxl_test)}")
            datasets.append(ptbxl_test)
        else:
            print(f"Warning: PTB-XL dataset not found at '{args.ptbxl_dir}'.")

    if args.dataset in ["all3", "both", "chapman"]:
        if os.path.exists(os.path.join(args.chapman_dir, "RECORDS")):
            print(f"Loading Chapman held-out test split (10% test fold)...")
            chapman_test = ChapmanECGDataset(data_dir=args.chapman_dir, split="test")
            print(f"  -> Chapman test records: {len(chapman_test)}")
            datasets.append(chapman_test)
        else:
            print(f"Warning: Chapman dataset not found at '{args.chapman_dir}'.")

    if args.dataset in ["all3", "mimic"]:
        mimic_csv = os.path.join(args.mimic_dir, "record_list.csv")
        if os.path.exists(mimic_csv):
            print(f"Loading MIMIC-IV-ECG held-out test split (10% patient-level fold)...")
            mimic_test = MIMICIVECGDataset(data_dir=args.mimic_dir, split="test")
            print(f"  -> MIMIC-IV-ECG test records: {len(mimic_test)}")
            datasets.append(mimic_test)
        else:
            print(f"Warning: MIMIC-IV-ECG dataset not found at '{args.mimic_dir}'.")
    if not datasets:
        print("Falling back to dummy test dataset...")
        test_dataset = DummyECGDataset(num_samples=args.max_samples or 100, num_channels=12, seq_length=5000)
    elif len(datasets) == 1:
        test_dataset = datasets[0]
    else:
        test_dataset = ConcatDataset(datasets)

    if args.max_samples is not None and len(test_dataset) > args.max_samples:
        test_dataset = Subset(test_dataset, list(range(args.max_samples)))
        print(f"Subsampled test dataset to {len(test_dataset)} records.")

    print(f"Total test records to evaluate: {len(test_dataset)}")
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 2. Initialize Model and Load Trained Weights
    print("Loading FoundationEncoder architecture and weights...")
    model = FoundationEncoder(
        in_channels=12,
        patch_size=args.patch_size,
        seq_length=5000,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=0.0,  # disable dropout during testing
        learned_pos=True,
    ).to(device)

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Evaluation Loop
    total_masked_sq_err = 0.0
    total_masked_elements = 0
    total_overall_sq_err = 0.0
    total_overall_elements = 0
    total_unmasked_sq_err = 0.0
    total_unmasked_elements = 0

    # Additional metrics accumulators
    total_masked_abs_err = 0.0        # for MAE
    # R² components (masked region)
    total_ss_res = 0.0                # sum of squared residuals
    total_ss_tot = 0.0                # sum of squared deviations from mean
    # Pearson correlation components (masked region)
    sum_orig = 0.0
    sum_recon = 0.0
    sum_orig_sq = 0.0
    sum_recon_sq = 0.0
    sum_cross = 0.0
    # SNR / PSNR
    total_signal_power = 0.0          # sum of orig^2 in masked region
    total_noise_power = 0.0           # sum of diff^2 in masked region (= masked_sq_err)

    # Per-lead metrics
    per_lead_sq_err = np.zeros(12, dtype=np.float64)
    per_lead_masked_counts = np.zeros(12, dtype=np.int64)

    first_batch_samples = []  # List to store up to num_plots samples

    print("\nRunning test evaluation...")
    with torch.no_grad():
        for batch_idx, original_signals in enumerate(dataloader):
            original_signals = original_signals.to(device) # (B, 12, 5000)
            B, C, L = original_signals.shape

            # Apply identical block masking as used in pre-training
            masked_signals, mask = apply_block_masking(
                original_signals,
                patch_size=args.patch_size,
                mask_ratio=args.mask_ratio
            ) # mask: (B, 1, 5000)

            # Reconstruct signal with trained model
            reconstructed_signals = model(masked_signals)

            # Collect one sample per batch up to num_plots for plotting
            if len(first_batch_samples) < args.num_plots:
                first_batch_samples.append((
                    original_signals[0].cpu().numpy(),
                    masked_signals[0].cpu().numpy(),
                    reconstructed_signals[0].cpu().numpy(),
                    mask[0].cpu().numpy()
                ))

            # Errors
            diff = reconstructed_signals - original_signals
            sq_diff = diff ** 2 # (B, 12, 5000)

            # Masked patch MSE (mask is now per-lead, (B, C, L), so no extra *C factor)
            masked_sq_diff = sq_diff * mask
            masked_elem_count = mask.sum().item()
            total_masked_sq_err += masked_sq_diff.sum().item()
            total_masked_elements += masked_elem_count

            # Overall sequence MSE
            total_overall_sq_err += sq_diff.sum().item()
            total_overall_elements += sq_diff.numel()

            # Unmasked patch MSE (fidelity on visible patches)
            unmasked_mask = 1.0 - mask
            unmasked_sq_diff = sq_diff * unmasked_mask
            unmasked_elem_count = unmasked_mask.sum().item()
            total_unmasked_sq_err += unmasked_sq_diff.sum().item()
            total_unmasked_elements += unmasked_elem_count

            # Per-lead masked squared errors (each lead uses its own mask count)
            for lead_idx in range(12):
                lead_err = masked_sq_diff[:, lead_idx, :].sum().item()
                lead_count = mask[:, lead_idx, :].sum().item()
                per_lead_sq_err[lead_idx] += lead_err
                per_lead_masked_counts[lead_idx] += int(lead_count)

            # ── Extra metrics on masked region ──────────────────────────────
            orig_masked  = (original_signals * mask)           # zero outside mask
            recon_masked = (reconstructed_signals * mask)
            diff_masked  = diff * mask

            # MAE
            total_masked_abs_err += diff_masked.abs().sum().item()

            # R²  (ss_res = sum sq residuals; ss_tot = sum sq dev from mean)
            orig_vals   = original_signals[mask > 0.5]         # flat 1-D
            recon_vals  = reconstructed_signals[mask > 0.5]
            orig_mean   = orig_vals.mean().item() if orig_vals.numel() > 0 else 0.0
            total_ss_res += ((orig_vals - recon_vals) ** 2).sum().item()
            total_ss_tot += ((orig_vals - orig_mean) ** 2).sum().item()

            # Pearson correlation running sums
            n_el = orig_vals.numel()
            sum_orig     += orig_vals.sum().item()
            sum_recon    += recon_vals.sum().item()
            sum_orig_sq  += (orig_vals ** 2).sum().item()
            sum_recon_sq += (recon_vals ** 2).sum().item()
            sum_cross    += (orig_vals * recon_vals).sum().item()

            # SNR / PSNR signal power
            total_signal_power += (original_signals[mask > 0.5] ** 2).sum().item()
            total_noise_power  += masked_sq_diff.sum().item()

    # 4. Compute Final Metrics
    masked_mse   = total_masked_sq_err  / max(total_masked_elements, 1)
    overall_mse  = total_overall_sq_err / max(total_overall_elements, 1)
    unmasked_mse = total_unmasked_sq_err / max(total_unmasked_elements, 1)
    masked_rmse  = np.sqrt(masked_mse)

    # MAE
    masked_mae = total_masked_abs_err / max(total_masked_elements, 1)

    # R²
    r_squared = 1.0 - (total_ss_res / max(total_ss_tot, 1e-12))

    # Pearson correlation (global, across all masked samples)
    N = total_masked_elements
    if N > 0:
        numerator = N * sum_cross - sum_orig * sum_recon
        denom_a   = max(N * sum_orig_sq  - sum_orig  ** 2, 0.0)
        denom_b   = max(N * sum_recon_sq - sum_recon ** 2, 0.0)
        denom     = np.sqrt(denom_a * denom_b)
        pearson_r = numerator / denom if denom > 1e-12 else 0.0
    else:
        pearson_r = 0.0

    # SNR  (dB) = 10 * log10(signal_power / noise_power)
    snr_db = 10.0 * np.log10(total_signal_power / max(total_noise_power, 1e-12))

    # PSNR (dB) — peak defined as max observed amplitude range (~2 for normalised)
    peak = 2.0
    psnr_db = 10.0 * np.log10((peak ** 2) / max(masked_mse, 1e-12))

    lead_mses = {}
    for i, name in enumerate(LEAD_NAMES):
        count = max(per_lead_masked_counts[i], 1)
        lead_mses[name] = float(per_lead_sq_err[i] / count)

    # 5. Display Clean Formatted Results
    print("\n" + "=" * 60)
    print("                PHASE 1 TEST EVALUATION RESULTS             ")
    print("=" * 60)
    print(f" PRIMARY METRIC (Masked Patch MSE) : {masked_mse:.6f}")
    print(f" Root Mean Squared Error (RMSE)    : {masked_rmse:.6f}")
    print(f" Mean Absolute Error  (MAE)        : {masked_mae:.6f}")
    print(f" R-Squared (R2)                    : {r_squared:.6f}")
    print(f" Pearson Correlation               : {pearson_r:.6f}")
    print(f" Signal-to-Noise Ratio (SNR)       : {snr_db:.2f} dB")
    print(f" Peak SNR (PSNR)                   : {psnr_db:.2f} dB")
    print(f" Overall Full-Sequence MSE         : {overall_mse:.6f}")
    print(f" Visible (Unmasked) Patch MSE      : {unmasked_mse:.6f}")
    print("-" * 60)
    print(" Per-Lead Masked Reconstruction MSE Breakdown:")
    for name, lmse in lead_mses.items():
        print(f"   Lead {name:<4} : MSE = {lmse:.6f}")
    print("=" * 60)

    # 6. Save JSON Report
    metrics_summary = {
        "checkpoint": args.checkpoint,
        "test_dataset": args.dataset,
        "total_test_records": len(test_dataset),
        "mask_ratio": args.mask_ratio,
        "masked_mse":    float(masked_mse),
        "masked_rmse":   float(masked_rmse),
        "masked_mae":    float(masked_mae),
        "r_squared":     float(r_squared),
        "pearson_r":     float(pearson_r),
        "snr_db":        float(snr_db),
        "psnr_db":       float(psnr_db),
        "overall_mse":   float(overall_mse),
        "unmasked_mse":  float(unmasked_mse),
        "per_lead_mse":  lead_mses,
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"Metrics saved to: {args.output_json}")

    # 7. Optional Plot Generation (Sample Waveform Comparison)
    if args.save_plot and len(first_batch_samples) > 0:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            for idx, (orig, masked, recon, mask_per_lead) in enumerate(first_batch_samples, start=1):
                # Prepare output path by formatting placeholder
                plot_path_template = args.save_plot
                if "{idx}" in plot_path_template:
                    plot_path = plot_path_template.replace("{idx}", str(idx))
                else:
                    # fallback: append index before extension
                    base, ext = os.path.splitext(plot_path_template)
                    plot_path = f"{base}_{idx}{ext}"
                plot_path = os.path.normpath(plot_path)
                plot_dir = os.path.dirname(plot_path)
                if plot_dir:
                    os.makedirs(plot_dir, exist_ok=True)

                fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
                time_axis = np.arange(orig.shape[1]) / 500.0  # seconds

                lead_idx = 1  # Lead II
                lead_label = LEAD_NAMES[lead_idx]
                m_mask = mask_per_lead[lead_idx]

                # 1. Original
                axes[0].plot(time_axis, orig[lead_idx], color="#1f77b4", linewidth=1.2,
                             label=f"Original (Lead {lead_label})")
                axes[0].set_title(f"1. Original Ground Truth ECG Signal (Lead {lead_label})", fontsize=11, fontweight="bold")
                axes[0].set_ylabel("Amplitude (normalized)")
                axes[0].grid(True, alpha=0.3)
                axes[0].legend(loc="upper right")

                # 2. Masked input
                axes[1].plot(time_axis, masked[lead_idx], color="#d62728", linewidth=1.2,
                             label=f"Masked Input ({args.mask_ratio*100:.0f}% block mask)")
                axes[1].fill_between(time_axis, -3, 3, where=(m_mask > 0.5),
                                     color="#ff9896", alpha=0.3, label="Masked span")
                axes[1].set_title("2. Masked Input Given to Model (60% Erased)", fontsize=11, fontweight="bold")
                axes[1].set_ylabel("Amplitude (normalized)")
                axes[1].grid(True, alpha=0.3)
                axes[1].legend(loc="upper right")

                # 3. Inpainted Composite
                inpainted = orig[lead_idx] * (1.0 - m_mask) + recon[lead_idx] * m_mask
                axes[2].plot(time_axis, orig[lead_idx], color="#1f77b4", alpha=0.4, linestyle="--",
                             label="Ground Truth")
                axes[2].plot(time_axis, inpainted, color="#ff7f0e", linewidth=1.3,
                             label="Inpainted ECG (Known + Model Inpainted)")
                axes[2].fill_between(time_axis, -3, 3, where=(m_mask > 0.5),
                                     color="#ff9896", alpha=0.15, label="Inpainted spans")
                axes[2].set_title("3. Inpainted ECG Output (True Known Spans + Model Predictions in Masked Spans)", fontsize=11, fontweight="bold")
                axes[2].set_ylabel("Amplitude (normalized)")
                axes[2].grid(True, alpha=0.3)
                axes[2].legend(loc="upper right")

                # 4. Raw Model Output vs Ground Truth
                axes[3].plot(time_axis, orig[lead_idx], color="#1f77b4", alpha=0.4, linestyle="--",
                             label="Ground Truth")
                axes[3].plot(time_axis, recon[lead_idx], color="#2ca02c", linewidth=1.3,
                             label="Raw Model Output")
                axes[3].set_title(f"4. Raw Model Output Across Entire Sequence (Masked MSE: {masked_mse:.4f})",
                                  fontsize=11, fontweight="bold")
                axes[3].set_xlabel("Time (seconds)")
                axes[3].set_ylabel("Amplitude (normalized)")
                axes[3].grid(True, alpha=0.3)
                axes[3].legend(loc="upper right")

                plt.tight_layout()
                fig.savefig(plot_path, dpi=200)
                plt.close(fig)
                plt.close('all')
                print(f"Sample waveform visualization saved to: {plot_path}")
        except Exception as e:
            print(f"Could not generate plot (skipped): {e}")

    return metrics_summary

if __name__ == "__main__":
    evaluate_phase1()
