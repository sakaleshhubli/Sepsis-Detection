"""
LSTM sequence model for the Sepsis Detection project.
Treats each patient's ICU stay as a sequence and predicts sepsis
probability at every timestep, so the model can learn temporal
deterioration patterns that single-timestep tree models can't see.
"""

import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from data_loader import load_and_validate
from features import engineer_features


# ---------- Sequence building ----------

def build_patient_sequences(df, feature_cols, label_col, id_col, time_col):
    """Group rows into per-patient (features, labels) sequences, sorted by time."""
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
    """Pad variable-length patient sequences within a batch."""
    seqs, labels, lengths = zip(*batch)
    lengths = torch.tensor(lengths)
    padded_seqs = pad_sequence(seqs, batch_first=True)      # (B, T_max, F)
    padded_labels = pad_sequence(labels, batch_first=True)  # (B, T_max)
    return padded_seqs, padded_labels, lengths


# ---------- Model ----------

class SepsisLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B, T_max, H)
        out = self.dropout(out)
        logits = self.fc(out).squeeze(-1)  # (B, T_max)
        return logits


# ---------- Training / evaluation ----------

def make_mask(lengths, max_len):
    """Boolean mask (B, T_max), True for valid (non-padded) timesteps."""
    return torch.arange(max_len)[None, :] < lengths[:, None]


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for seqs, labels, lengths in loader:
        seqs, labels = seqs.to(device), labels.to(device)
        mask = make_mask(lengths, seqs.size(1)).to(device)

        optimizer.zero_grad()
        logits = model(seqs, lengths)
        loss = criterion(logits, labels)
        loss = (loss * mask).sum() / mask.sum()
        loss.backward()
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
        mask = make_mask(lengths, seqs.size(1)).to(device)

        logits = model(seqs, lengths)
        loss = criterion(logits, labels)
        loss = (loss * mask).sum() / mask.sum()
        total_loss += loss.item() * seqs.size(0)

        probs = torch.sigmoid(logits)
        mask_np = mask.cpu().numpy().astype(bool)
        all_probs.append(probs.cpu().numpy()[mask_np])
        all_labels.append(labels.cpu().numpy()[mask_np])

    avg_loss = total_loss / len(loader.dataset)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)
    return avg_loss, auroc, auprc


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df, val_df, test_df, config = load_and_validate()

    print("\nEngineering features...")
    train_feat, fill_values = engineer_features(train_df, config)
    val_feat, _ = engineer_features(val_df, config, fill_values=fill_values)
    test_feat, _ = engineer_features(test_df, config, fill_values=fill_values)

    id_col = config["columns"]["id_col"]
    time_col = config["columns"]["time_col"]
    label_col = config["columns"]["label_col"]
    exclude = {id_col, time_col, label_col}
    feature_cols = [c for c in train_feat.columns if c not in exclude]
    print(f"Using {len(feature_cols)} features, sequence modeling over patient history")

    # Standardize features using train statistics only (fit on train, apply to val/test)
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
    pos_weight = torch.tensor(
        (len(all_train_labels) - all_train_labels.sum()) / all_train_labels.sum(),
        dtype=torch.float32,
    ).to(device)
    print(f"pos_weight: {pos_weight.item():.2f}")

    model = SepsisLSTM(input_size=len(feature_cols), hidden_size=64, num_layers=2, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    best_val_auprc = 0.0
    patience, patience_counter = 5, 0
    model_path = os.path.join(config["paths"]["model_dir"], "sepsis_lstm.pt")
    os.makedirs(config["paths"]["model_dir"], exist_ok=True)

    for epoch in range(1, 31):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auroc, val_auprc = evaluate_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch:2d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_auroc={val_auroc:.4f} | val_auprc={val_auprc:.4f}")

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

    with open(os.path.join(config["paths"]["model_dir"], "lstm_meta.pkl"), "wb") as f:
        pickle.dump({"scaler": scaler, "feature_cols": feature_cols, "input_size": len(feature_cols)}, f)
    print("Saved scaler + metadata to models/lstm_meta.pkl")


if __name__ == "__main__":
    main()