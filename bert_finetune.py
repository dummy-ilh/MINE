"""
Practical BERT fine-tuning script for text classification.
Every knob discussed in the companion docs (learning rate, warmup, layer
freezing, discriminative LR, class weighting, gradient accumulation, early
stopping) is exposed as a single CONFIG object at the top -- change values
there, nothing else needs to move.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)


# ---------------------------------------------------------------------------
# 1. ALL TUNABLE ITEMS LIVE HERE. This is the only section you should need
#    to touch to run a different experiment.
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # --- model / data ---
    model_name: str = "bert-base-uncased"
    num_labels: int = 2
    max_length: int = 128                     # set from your data's 95th/99th pct token length

    # --- optimization ---
    learning_rate: float = 2e-5                # 2e-5 - 5e-5 is the standard fine-tuning range
    head_learning_rate: float = 1e-3            # new head can use a much larger LR than the encoder
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1                   # fraction of total steps spent warming up
    layer_lr_decay: float = 0.9                 # discriminative LR multiplier per layer, going down from top

    # --- batching ---
    per_device_batch_size: int = 16
    gradient_accumulation_steps: int = 2         # effective_batch = per_device_bs * this
    num_epochs: int = 4
    max_grad_norm: float = 1.0                   # gradient clipping

    # --- layer freezing ---
    num_layers_to_unfreeze: int = 4              # 0 = freeze whole encoder (head-only training)
                                                  # 12 = full fine-tuning (BERT-base has 12 layers)
    freeze_embeddings: bool = True               # embeddings are almost always safe to freeze

    # --- regularization / imbalance ---
    dropout: float = 0.1
    class_weights: Optional[List[float]] = None  # e.g. [0.56, 5.0] for a 900:100 imbalance; None = unweighted

    # --- early stopping ---
    early_stopping_patience: int = 2             # stop if val loss doesn't improve for N eval rounds
    eval_every_n_steps: int = 100

    seed: int = 42


CONFIG = Config()


# ---------------------------------------------------------------------------
# 2. Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 3. Dataset
# ---------------------------------------------------------------------------
class ClassificationDataset(Dataset):
    """texts: List[str], labels: List[int]"""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# 4. Model: BERT encoder + a small classification head
# ---------------------------------------------------------------------------
class BertClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_vector = out.last_hidden_state[:, 0, :]   # [CLS] token, shape (B, hidden_size)
        cls_vector = self.dropout(cls_vector)
        logits = self.classifier(cls_vector)            # (B, num_labels)
        return logits


# ---------------------------------------------------------------------------
# 5. Layer freezing -- freeze embeddings + bottom (12 - num_unfrozen) layers,
#    leave the top `num_layers_to_unfreeze` layers + head trainable.
# ---------------------------------------------------------------------------
def apply_layer_freezing(model: BertClassifier, cfg: Config):
    if cfg.freeze_embeddings:
        for p in model.bert.embeddings.parameters():
            p.requires_grad = False

    encoder_layers = model.bert.encoder.layer          # ModuleList of 12 BertLayer blocks
    total_layers = len(encoder_layers)
    num_frozen = max(0, total_layers - cfg.num_layers_to_unfreeze)

    for i, layer in enumerate(encoder_layers):
        trainable = i >= num_frozen
        for p in layer.parameters():
            p.requires_grad = trainable

    # classifier head is always trainable
    for p in model.classifier.parameters():
        p.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[freezing] {num_frozen}/{total_layers} encoder layers frozen "
          f"(+ embeddings {'frozen' if cfg.freeze_embeddings else 'trainable'}) | "
          f"trainable params: {n_trainable:,} / {n_total:,} "
          f"({100 * n_trainable / n_total:.1f}%)")


# ---------------------------------------------------------------------------
# 6. Discriminative learning rates -- higher LR for the head, decaying LR
#    for the encoder as you go deeper (layer 12 fastest, layer 1 slowest).
#    Frozen params are skipped automatically since requires_grad is False.
# ---------------------------------------------------------------------------
def build_optimizer_param_groups(model: BertClassifier, cfg: Config):
    encoder_layers = model.bert.encoder.layer
    total_layers = len(encoder_layers)
    no_decay = ["bias", "LayerNorm.weight"]

    param_groups = []

    # head -- largest LR, since it's randomly initialized and needs to move the most
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr": cfg.head_learning_rate,
            "weight_decay": cfg.weight_decay,
        })

    # encoder layers -- LR decays going from top (layer 12) to bottom (layer 1)
    for i, layer in enumerate(encoder_layers):
        depth_from_top = total_layers - 1 - i          # 0 for the last layer, grows going down
        layer_lr = cfg.learning_rate * (cfg.layer_lr_decay ** depth_from_top)

        decay_params = [p for n, p in layer.named_parameters()
                         if p.requires_grad and not any(nd in n for nd in no_decay)]
        no_decay_params = [p for n, p in layer.named_parameters()
                            if p.requires_grad and any(nd in n for nd in no_decay)]

        if decay_params:
            param_groups.append({"params": decay_params, "lr": layer_lr, "weight_decay": cfg.weight_decay})
        if no_decay_params:
            param_groups.append({"params": no_decay_params, "lr": layer_lr, "weight_decay": 0.0})

    return param_groups


# ---------------------------------------------------------------------------
# 7. Training loop
# ---------------------------------------------------------------------------
def train(model, train_loader, val_loader, cfg: Config, device):
    param_groups = build_optimizer_param_groups(model, cfg)
    optimizer = torch.optim.AdamW(param_groups)

    steps_per_epoch = math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
    total_steps = steps_per_epoch * cfg.num_epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if cfg.class_weights is not None:
        weight_tensor = torch.tensor(cfg.class_weights, dtype=torch.float, device=device)
    else:
        weight_tensor = None
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    model.to(device)
    for epoch in range(cfg.num_epochs):
        model.train()
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids"))
            loss = criterion(logits, batch["labels"]) / cfg.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % cfg.eval_every_n_steps == 0:
                    val_loss = evaluate(model, val_loader, criterion, device)
                    model.train()
                    print(f"epoch {epoch} step {global_step} | val_loss {val_loss:.4f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # torch.save(model.state_dict(), "best_model.pt")
                    else:
                        patience_counter += 1
                        if patience_counter >= cfg.early_stopping_patience:
                            print("[early stopping] validation loss stopped improving.")
                            return

    print("Training complete.")


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids"))
        loss = criterion(logits, batch["labels"])
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# 8. Wire it together
# ---------------------------------------------------------------------------
def main():
    set_seed(CONFIG.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG.model_name)
    model = BertClassifier(CONFIG.model_name, CONFIG.num_labels, CONFIG.dropout)
    apply_layer_freezing(model, CONFIG)

    # --- replace with your real data ---
    train_texts, train_labels = ["example sentence one", "example sentence two"], [0, 1]
    val_texts, val_labels = ["a validation example"], [0]

    train_ds = ClassificationDataset(train_texts, train_labels, tokenizer, CONFIG.max_length)
    val_ds = ClassificationDataset(val_texts, val_labels, tokenizer, CONFIG.max_length)

    train_loader = DataLoader(train_ds, batch_size=CONFIG.per_device_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG.per_device_batch_size, shuffle=False)

    train(model, train_loader, val_loader, CONFIG, device)


if __name__ == "__main__":
    main()
