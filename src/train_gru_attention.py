"""
GRU + causal self-attention sequence model for the Sepsis Detection project.

Improves on train_lstm.py in two ways:
  1. Uses the full PhysioNet feature set (8 vitals, 26 labs, 5 static —
     set via configs/config.yaml) instead of the 17-column subset.
  2. Adds a causal multi-head self-attention layer on top of the GRU
     outputs, so the prediction at hour t can directly attend back to
     whichever earlier hours were most informative, rather than relying
     only on the GRU's single compressed hidden state. Attention is
     masked to be causal (no peeking at future hours) and padding-aware
     (no attending to padded timesteps), since this must stay usable
     for real-time, hour-by-hour prediction.

Run: python src/train_gru_attention.py
Saves: models/sepsis_gru_attn.pt, models/gru_attn_meta.pkl
"""

import os
import math
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from data_loader import load_and_validate
from features import engineer_features


# ---------- Sequence building (same convention as train_lstm.py) ----------

def build_patient_sequences(df, feature_cols, label_col, id_col, time_col):
    df = df.sort_values([id_col, time_col])
    sequences, labels = [], []
    for _, group in df.groupby(id_col):
        sequences.append(group[feature_cols].values.astype(np.float32))
        labels.append(group[label_col].values.astype(np.float32))
    return sequences, labels


class SepsisSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx]),
            torch.tensor(self.labels[idx]),
            len(self.sequences[idx]),
        )


def collate_fn(batch):
    seqs, labels, lengths = zip(*batch)
    lengths = torch.tensor(lengths)
    padded_seqs = pad_sequence(seqs, batch_first=True)      # (B, T_max, F)
    padded_labels = pad_sequence(labels, batch_first=True)  # (B, T_max)
    return padded_seqs, padded_labels, lengths


def make_mask(lengths, max_len):
    """Boolean mask (B, T_max), True for valid (non-padded) timesteps."""
    return torch.arange(max_len)[None, :] < lengths[:, None]


# ---------- Model ----------

class GRUAttentionSepsis(nn.Module):
    def __init__(self, input_size, hidden_size=96, gru_layers=2,
                 attn_heads=4, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers=gru_layers,
            batch_first=True, dropout=dropout if gru_layers > 1 else 0,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=attn_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    @staticmethod
    def _causal_mask(seq_len, device):
        # True = disallowed position (can't attend to the future)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        return mask

    def forward(self, x, lengths, pad_mask):
        """
        x: (B, T, F) padded input
        lengths: (B,) true sequence lengths
        pad_mask: (B, T) True for valid (non-padded) timesteps
        """
        gru_out, _ = self.gru(x)  # (B, T, H) — GRU itself ignores padding correctness
        # ok as-is for our purposes; padded steps are masked out of loss and attention

        T = gru_out.size(1)
        causal = self._causal_mask(T, x.device)                     # (T, T) bool, True = blocked
        key_padding_mask = ~pad_mask                                 # (B, T) True = ignore (padded)

        attn_out, _ = self.attn(
            gru_out, gru_out, gru_out,
            attn_mask=causal,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        fused = self.norm(gru_out + attn_out)   # residual connection + LayerNorm
        fused = self.dropout(fused)
        logits = self.fc(fused).squeeze(-1)     # (B, T)
        return logits


# ---------- Training / evaluation ----------

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for seqs, labels, lengths in loader:
        seqs, labels = seqs.to(device), labels.to(device)
        pad_mask = make_mask(lengths, seqs.size(1)).to(device)

        optimizer.zero_grad()
        logits = model(seqs, lengths, pad_mask)
        loss = criterion(logits, labels)
        loss = (loss * pad_mask).sum() / pad_mask.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * seqs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    for seqs, labels, lengths in loader:
        seqs, labels = seqs.to(device), labels.to(device)
        pad_mask = make_mask(lengths, seqs.size(1)).to(device)

        logits = model(seqs, lengths, pad_mask)
        loss = criterion(logits, labels)
        loss = (loss * pad_mask).sum() / pad_mask.sum()
        total_loss += loss.item() * seqs.size(0)

        probs = torch.sigmoid(logits)
        mask_np = pad_mask.cpu().numpy().astype(bool)
        all_probs.append(probs.cpu().numpy()[mask_np])
        all_labels.append(labels.cpu().numpy()[mask_np])

    avg_loss = total_loss / len(loader.dataset)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)
    return avg_loss, auroc, auprc


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    train_df, val_df, test_df, config = load_and_validate()

    print("\nEngineering features (full column set from config.yaml)...")
    train_feat, fill_values = engineer_features(train_df, config)
    val_feat, _ = engineer_features(val_df, config, fill_values=fill_values)
    test_feat, _ = engineer_features(test_df, config, fill_values=fill_values)

    id_col = config["columns"]["id_col"]
    time_col = config["columns"]["time_col"]
    label_col = config["columns"]["label_col"]
    exclude = {id_col, time_col, label_col}
    feature_cols = [c for c in train_feat.columns if c not in exclude]
    print(f"Using {len(feature_cols)} features, sequence modeling over patient history")

    scaler = StandardScaler()
    train_feat[feature_cols] = scaler.fit_transform(train_feat[feature_cols])
    val_feat[feature_cols] = scaler.transform(val_feat[feature_cols])
    test_feat[feature_cols] = scaler.transform(test_feat[feature_cols])

    train_seqs, train_labels = build_patient_sequences(train_feat, feature_cols, label_col, id_col, time_col)
    val_seqs, val_labels = build_patient_sequences(val_feat, feature_cols, label_col, id_col, time_col)
    test_seqs, test_labels = build_patient_sequences(test_feat, feature_cols, label_col, id_col, time_col)

    train_loader = DataLoader(SepsisSequenceDataset(train_seqs, train_labels), batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(SepsisSequenceDataset(val_seqs, val_labels), batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(SepsisSequenceDataset(test_seqs, test_labels), batch_size=64, shuffle=False, collate_fn=collate_fn)

    all_train_labels = np.concatenate(train_labels)
    raw_pos_weight = (len(all_train_labels) - all_train_labels.sum()) / all_train_labels.sum()
    # Softened via sqrt: the raw ratio (~54x) overcorrects on every rare-positive
    # example and was driving sharp val-loss spikes after a few epochs.
    pos_weight = torch.tensor(math.sqrt(raw_pos_weight), dtype=torch.float32).to(device)
    print(f"raw pos_weight: {raw_pos_weight:.2f} | used (sqrt-softened): {pos_weight.item():.2f}")

    model = GRUAttentionSepsis(
        input_size=len(feature_cols), hidden_size=64, gru_layers=2,
        attn_heads=2, dropout=0.5,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    best_val_auprc = 0.0
    patience, patience_counter = 5, 0
    model_path = os.path.join(config["paths"]["model_dir"], "sepsis_gru_attn.pt")
    os.makedirs(config["paths"]["model_dir"], exist_ok=True)

    for epoch in range(1, 31):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auroc, val_auprc = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_auprc)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:2d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_auroc={val_auroc:.4f} | val_auprc={val_auprc:.4f} | lr={current_lr:.2e}")

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(model_path))
    _, train_auroc, train_auprc = evaluate_epoch(model, train_loader, criterion, device)
    _, val_auroc, val_auprc = evaluate_epoch(model, val_loader, criterion, device)
    _, test_auroc, test_auprc = evaluate_epoch(model, test_loader, criterion, device)

    print(f"\nFinal — Train AUROC: {train_auroc:.4f} | AUPRC: {train_auprc:.4f}")
    print(f"Final — Val   AUROC: {val_auroc:.4f} | AUPRC: {val_auprc:.4f}")
    print(f"Final — Test  AUROC: {test_auroc:.4f} | AUPRC: {test_auprc:.4f}")

    with open(os.path.join(config["paths"]["model_dir"], "gru_attn_meta.pkl"), "wb") as f:
        pickle.dump(
            {"scaler": scaler, "feature_cols": feature_cols, "input_size": len(feature_cols)},
            f,
        )
    print("Saved scaler + metadata to models/gru_attn_meta.pkl")


if __name__ == "__main__":
    main()