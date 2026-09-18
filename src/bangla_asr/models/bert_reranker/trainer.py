"""
PyTorch training loop and validation for BanglaBERTReranker.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from bangla_asr.common.seed import set_seed
from bangla_asr.config import BERTRerankerConfig
from bangla_asr.dataset.splitter import extract_source_id
from bangla_asr.models.bert_reranker.dataset import CandidatePairDataset
from bangla_asr.models.bert_reranker.model import BanglaBERTReranker


def _split_records_group_aware(
    records: List[Dict],
    val_split_frac: float,
    seed: int,
) -> tuple:
    groups: Dict[str, List[Dict]] = {}
    for rec in records:
        gid = extract_source_id(rec.get("audio_path") or rec.get("utt_id", "unknown"))
        groups.setdefault(gid, []).append(rec)

    keys = list(groups.keys())
    generator = torch.Generator().manual_seed(seed)
    if len(keys) < 2:
        permutation = torch.randperm(len(records), generator=generator).tolist()
        n_val = min(len(records) - 1, max(1, round(len(records) * val_split_frac)))
        val_idx = set(permutation[:n_val])
        train = [r for i, r in enumerate(records) if i not in val_idx]
        val = [r for i, r in enumerate(records) if i in val_idx]
        return train, val

    perm = torch.randperm(len(keys), generator=generator).tolist()
    n_val = min(len(keys) - 1, max(1, round(len(keys) * val_split_frac)))
    val_keys = {keys[i] for i in perm[:n_val]}
    train, val = [], []
    for key, recs in groups.items():
        (val if key in val_keys else train).extend(recs)
    return train, val


def train_bert_reranker(
    train_records: List[Dict],
    output_dir: str = "checkpoints/bert_reranker",
    config: BERTRerankerConfig = BERTRerankerConfig(),
    val_split_frac: float = 0.1,
    device: Optional[str] = None,
) -> str:
    """
    Trains BanglaBERTReranker on candidate pool instances to predict candidate WER.
    Saves the best model weights and tokenizer to output_dir.
    """
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    device_obj = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(config.seed)
    print(f"Initializing BERT Reranker Training on {device_obj}...")

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    usable_records = [record for record in train_records if record.get("candidates")]
    if len(usable_records) < 2:
        raise ValueError("At least two N-best records with candidates are required for training and validation")

    if not 0.0 < val_split_frac < 1.0:
        raise ValueError("val_split_frac must be strictly between 0 and 1")

    # Split by source recording before flattening candidates so beams from the
    # same audio/reference cannot leak into both train and validation.
    train_records_split, val_records_split = _split_records_group_aware(
        usable_records, val_split_frac, config.seed
    )
    train_set = CandidatePairDataset(train_records_split, tokenizer, config=config, is_training=True)
    val_set = CandidatePairDataset(val_records_split, tokenizer, config=config, is_training=True)
    if not train_set or not val_set:
        raise ValueError("Training and validation splits must both contain at least one candidate")

    pin_memory = device_obj.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    model = BanglaBERTReranker(config).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.SmoothL1Loss()

    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    best_checkpoint = save_path / "best_model.pt"
    epochs_without_improve = 0

    print(f"Dataset: {len(train_set)} train candidates, {len(val_set)} val candidates.")

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device_obj)
            attention_mask = batch["attention_mask"].to(device_obj)
            acoustic = batch["acoustic_features"].to(device_obj) if config.use_acoustic_features else None
            targets = batch["target_wer"].to(device_obj)

            optimizer.zero_grad()
            preds = model(input_ids, attention_mask, acoustic)
            loss = criterion(preds, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device_obj)
                attention_mask = batch["attention_mask"].to(device_obj)
                acoustic = batch["acoustic_features"].to(device_obj) if config.use_acoustic_features else None
                targets = batch["target_wer"].to(device_obj)

                preds = model(input_ids, attention_mask, acoustic)
                loss = criterion(preds, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / max(1, len(val_loader))
        print(f"Epoch {epoch}/{config.num_epochs} — Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improve = 0
            torch.save({"model_state_dict": model.state_dict()}, best_checkpoint)
            tokenizer.save_pretrained(str(save_path))
            with open(save_path / "reranker_config.json", "w", encoding="utf-8") as f:
                json.dump(asdict(config), f, ensure_ascii=False, indent=2)
            print(f"  --> Saved new best checkpoint to {best_checkpoint}")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= config.early_stopping_patience:
                print(
                    f"Early stopping after {epoch} epochs "
                    f"({config.early_stopping_patience} epochs without val improvement)."
                )
                break

    print(f"Training complete. Best Validation Loss: {best_val_loss:.4f}")
    return str(best_checkpoint)
