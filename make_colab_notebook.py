import json

def code_cell(src):
    return {"cell_type": "code", "metadata": {}, "source": src, "outputs": [], "execution_count": None}

def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

cells = []

# ── Cell 0: Title ───────────────────────────────────────────────────────────
cells.append(md_cell(
    "# FYP Phase 1 – ECG Foundation Encoder (Google Colab T4 GPU)\n"
    "Self-supervised masked reconstruction pre-training on PTB-XL.\n\n"
    "**Runtime → Change runtime type → T4 GPU** before running."
))

# ── Cell 1: Check GPU ───────────────────────────────────────────────────────
cells.append(code_cell(
    "import torch\n"
    "print('GPU available:', torch.cuda.is_available())\n"
    "if torch.cuda.is_available():\n"
    "    print('Device:', torch.cuda.get_device_name(0))\n"
    "!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"
))

# ── Cell 2: Install dependencies ────────────────────────────────────────────
cells.append(code_cell("!pip install -q wfdb pandas numpy matplotlib"))

# ── Cell 3: Download PTB-XL ─────────────────────────────────────────────────
cells.append(code_cell(
    "import os\n"
    "PTBXL_DIR = '/content/data/ptbxl'\n"
    "os.makedirs(PTBXL_DIR, exist_ok=True)\n"
    "\n"
    "if not os.path.exists(f'{PTBXL_DIR}/ptbxl_database.csv'):\n"
    "    print('Downloading PTB-XL (~2 GB) from PhysioNet...')\n"
    "    !wget -q -r -N -c -np --cut-dirs=3 -P {PTBXL_DIR} \\\n"
    "        https://physionet.org/files/ptb-xl/1.0.3/\n"
    "    print('Download complete!')\n"
    "else:\n"
    "    print('PTB-XL already downloaded.')"
))

# ── Cell 4: Create directory structure ──────────────────────────────────────
cells.append(code_cell(
    "import os\n"
    "for d in ['/content/src/models', '/content/src/data',\n"
    "          '/content/src/training', '/content/checkpoints', '/content/reports']:\n"
    "    os.makedirs(d, exist_ok=True)\n"
    "for d in ['/content/src', '/content/src/models', '/content/src/data']:\n"
    "    open(f'{d}/__init__.py', 'w').close()\n"
    "print('Directory structure ready.')"
))

# ── Cell 5: Write masking.py ─────────────────────────────────────────────────
masking_code = r"""%%writefile /content/src/data/masking.py
import torch, random

def apply_block_masking(signal, patch_size=50, mask_ratio=0.6, min_span=2, max_span=6):
    if signal.dim() == 2:
        signal = signal.unsqueeze(0)
    B, C, L = signal.shape
    num_patches = L // patch_size
    target_masked = int(round(num_patches * mask_ratio))
    patch_masks = []
    for b in range(B):
        mask_indices = set()
        attempts = 0
        while len(mask_indices) < target_masked and attempts < 1000:
            attempts += 1
            span_len = random.randint(min_span, max_span)
            start_idx = random.randint(0, max(0, num_patches - span_len))
            for i in range(start_idx, min(num_patches, start_idx + span_len)):
                mask_indices.add(i)
                if len(mask_indices) >= target_masked:
                    break
        if len(mask_indices) < target_masked:
            unmasked = [i for i in range(num_patches) if i not in mask_indices]
            needed = target_masked - len(mask_indices)
            for i in random.sample(unmasked, min(needed, len(unmasked))):
                mask_indices.add(i)
        b_mask = torch.zeros(num_patches, dtype=torch.float32, device=signal.device)
        if mask_indices:
            b_mask[list(mask_indices)] = 1.0
        patch_masks.append(b_mask)
    patch_mask = torch.stack(patch_masks, dim=0)
    full_mask = patch_mask.unsqueeze(-1).repeat(1, 1, patch_size).view(B, -1)
    if full_mask.shape[1] < L:
        pad = torch.zeros(B, L - full_mask.shape[1], device=signal.device)
        full_mask = torch.cat([full_mask, pad], dim=1)
    full_mask = full_mask.unsqueeze(1)
    return signal * (1.0 - full_mask), full_mask
"""
cells.append(code_cell(masking_code))

# ── Cell 6: Write ptbxl_dataset.py ──────────────────────────────────────────
ptbxl_code = r"""%%writefile /content/src/data/ptbxl_dataset.py
import os, torch, wfdb
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class PTBXLECGDataset(Dataset):
    def __init__(self, data_dir='data/ptbxl', sampling_rate=500, split='train',
                 test_fold=10, val_fold=9, normalize=True, max_samples=None):
        self.data_dir = data_dir
        self.sampling_rate = sampling_rate
        self.normalize = normalize
        df = pd.read_csv(os.path.join(data_dir, 'ptbxl_database.csv'))
        if split == 'train':
            df = df[(df.strat_fold != test_fold) & (df.strat_fold != val_fold)]
        elif split == 'val':
            df = df[df.strat_fold == val_fold]
        elif split == 'test':
            df = df[df.strat_fold == test_fold]
        if max_samples:
            df = df.iloc[:max_samples]
        self.df = df.reset_index(drop=True)
        self.filename_col = 'filename_hr' if sampling_rate == 500 else 'filename_lr'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        record_path = os.path.join(self.data_dir, self.df.iloc[idx][self.filename_col])
        try:
            signal, _ = wfdb.rdsamp(record_path)
        except Exception:
            alt = (idx + 1) % len(self.df)
            signal, _ = wfdb.rdsamp(os.path.join(self.data_dir, self.df.iloc[alt][self.filename_col]))
        if np.isnan(signal).any():
            signal = np.nan_to_num(signal, nan=0.0)
        signal = signal.T.astype(np.float32)
        if self.normalize:
            mean = np.mean(signal, axis=1, keepdims=True)
            std  = np.std(signal, axis=1, keepdims=True) + 1e-8
            signal = (signal - mean) / std
        return torch.from_numpy(signal)
"""
cells.append(code_cell(ptbxl_code))

# ── Cell 7: Write foundation_encoder.py ─────────────────────────────────────
encoder_code = r"""%%writefile /content/src/models/foundation_encoder.py
import torch, torch.nn as nn, math

class FoundationEncoder(nn.Module):
    def __init__(self, in_channels=12, patch_size=50, embed_dim=256, num_layers=8,
                 num_heads=8, seq_length=5000, dropout=0.1, learned_pos=True):
        super().__init__()
        self.patch_size  = patch_size
        self.in_channels = in_channels
        assert seq_length % patch_size == 0
        self.num_patches = seq_length // patch_size
        self.patch_embed  = nn.Conv1d(in_channels, embed_dim, patch_size, patch_size)
        self.pos_embed    = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop     = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim*4,
                                               dropout, 'gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers, enable_nested_tensor=False)
        self.norm              = nn.LayerNorm(embed_dim)
        self.reconstruction_head = nn.Linear(embed_dim, in_channels * patch_size)

    def forward_features(self, x):
        x = self.patch_embed(x).transpose(1, 2) + self.pos_embed
        return self.norm(self.transformer(self.pos_drop(x)))

    def forward(self, x):
        B = x.size(0)
        out = self.reconstruction_head(self.forward_features(x))
        N   = out.size(1)
        return out.view(B, N, self.in_channels, self.patch_size).permute(0,2,1,3).reshape(B, self.in_channels, -1)
"""
cells.append(code_cell(encoder_code))

# ── Cell 8: Training ─────────────────────────────────────────────────────────
train_code = """import sys, os, json, torch, torch.nn as nn, torch.optim as optim, numpy as np
sys.path.insert(0, '/content/src')
from torch.utils.data import DataLoader
from data.ptbxl_dataset import PTBXLECGDataset
from data.masking import apply_block_masking
from models.foundation_encoder import FoundationEncoder

# ── Config ────────────────────────────────────────────────────────────────────
PTBXL_DIR   = '/content/data/ptbxl'
BATCH_SIZE  = 64        # GPU can handle larger batches
EPOCHS      = 10
LR          = 2e-4
PATCH_SIZE  = 50
MASK_RATIO  = 0.6
BEST_PATH   = '/content/checkpoints/foundation_encoder_best.pth'
SAVE_PATH   = '/content/checkpoints/foundation_encoder_phase1.pth'
MAX_SAMPLES = None      # set e.g. 3000 for a fast smoke-test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ── Data ──────────────────────────────────────────────────────────────────────
print('Loading PTB-XL...')
train_ds = PTBXLECGDataset(PTBXL_DIR, split='train', max_samples=MAX_SAMPLES)
val_ds   = PTBXLECGDataset(PTBXL_DIR, split='val')
print(f'Train: {len(train_ds):,}  Val: {len(val_ds):,}')
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ── Model ─────────────────────────────────────────────────────────────────────
model = FoundationEncoder().to(device)
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

decay     = [p for n,p in model.named_parameters() if 'bias' not in n and 'norm' not in n and 'pos_embed' not in n]
no_decay  = [p for n,p in model.named_parameters() if 'bias' in n or 'norm' in n or 'pos_embed' in n]
optimizer = optim.AdamW([{'params': decay, 'weight_decay': 0.05},
                          {'params': no_decay, 'weight_decay': 0.0}], lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*EPOCHS, eta_min=1e-6)

best_val, history = float('inf'), []
print(f'\\nStarting Phase 1 training — {EPOCHS} epochs, mask_ratio={MASK_RATIO}\\n')

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for sig in train_loader:
        sig = sig.to(device)
        masked, mask = apply_block_masking(sig, PATCH_SIZE, MASK_RATIO)
        optimizer.zero_grad()
        recon = model(masked)
        loss  = ((recon - sig)**2 * mask).sum() / mask.sum().clamp(min=1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for sig in val_loader:
            sig = sig.to(device)
            m, mk = apply_block_masking(sig, PATCH_SIZE, MASK_RATIO)
            v = model(m)
            val_loss += ((v - sig)**2 * mk).sum().item() / mk.sum().clamp(min=1.0).item()
    val_loss /= len(val_loader)
    lr_now = scheduler.get_last_lr()[0]
    history.append({'epoch': epoch+1, 'train': train_loss, 'val': val_loss})
    print(f'Epoch [{epoch+1:2d}/{EPOCHS}]  Train: {train_loss:.6f}  Val: {val_loss:.6f}  LR: {lr_now:.2e}')
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), BEST_PATH)
        print(f'   -> Best checkpoint saved (Val: {best_val:.6f})')

torch.save(model.state_dict(), SAVE_PATH)
print('\\nTraining complete!')
with open('/content/reports/train_history.json', 'w') as f:
    json.dump(history, f, indent=2)
"""
cells.append(code_cell(train_code))

# ── Cell 9: Test Evaluation ──────────────────────────────────────────────────
eval_code = """LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
test_ds     = PTBXLECGDataset(PTBXL_DIR, split='test')
test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
print(f'Test records: {len(test_ds):,}')

model.load_state_dict(torch.load(BEST_PATH, map_location=device))
model.eval()
tot_sq, tot_n = 0.0, 0
pl_sq = np.zeros(12); pl_n = np.zeros(12)

with torch.no_grad():
    for sig in test_loader:
        sig = sig.to(device)
        masked, mask = apply_block_masking(sig, PATCH_SIZE, MASK_RATIO)
        recon = model(masked)
        diff  = (recon - sig)**2
        tot_sq += (diff * mask).sum().item()
        tot_n  += mask.sum().item() * 12
        for li in range(12):
            pl_sq[li] += (diff[:,li,:] * mask[:,0,:]).sum().item()
            pl_n[li]  += mask.sum().item()

mse  = tot_sq / max(tot_n, 1)
rmse = mse ** 0.5
lmse = {LEAD_NAMES[i]: float(pl_sq[i]/max(pl_n[i],1)) for i in range(12)}

print('\\n' + '='*55)
print('  PHASE 1 TEST RESULTS (PTB-XL)')
print('='*55)
print(f'  Masked MSE  : {mse:.6f}')
print(f'  Masked RMSE : {rmse:.6f}')
print('-'*55)
for n, v in lmse.items():
    print(f'  Lead {n:<4}: {v:.6f}')
print('='*55)

metrics = {'masked_mse': mse, 'masked_rmse': rmse, 'test_records': len(test_ds), 'per_lead_mse': lmse}
with open('/content/reports/eval_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print('Saved: /content/reports/eval_metrics.json')
"""
cells.append(code_cell(eval_code))

# ── Cell 10: Plot training curve ─────────────────────────────────────────────
plot_code = """import matplotlib.pyplot as plt, json
with open('/content/reports/train_history.json') as f:
    h = json.load(f)
epochs = [x['epoch'] for x in h]
plt.figure(figsize=(10, 4))
plt.plot(epochs, [x['train'] for x in h], 'b-o', label='Train Loss')
plt.plot(epochs, [x['val']   for x in h], 'r-o', label='Val Loss')
plt.xlabel('Epoch'); plt.ylabel('Masked MSE Loss')
plt.title('Phase 1 Training Curve — PTB-XL'); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig('/content/reports/training_curve.png', dpi=150)
plt.show()
print('Plot saved.')
"""
cells.append(code_cell(plot_code))

# ── Cell 11: Download outputs ────────────────────────────────────────────────
cells.append(code_cell(
    "from google.colab import files\n"
    "files.download('/content/checkpoints/foundation_encoder_best.pth')\n"
    "files.download('/content/reports/eval_metrics.json')\n"
    "files.download('/content/reports/training_curve.png')\n"
    "print('Files downloaded!')"
))

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "accelerator": "GPU"
    },
    "cells": cells
}

with open(r"d:\Datasets\FYP\FYP_Phase1_Colab.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("SUCCESS: FYP_Phase1_Colab.ipynb created at d:\\Datasets\\FYP\\")
