import os
import json
import random
from typing import List, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from app.data.training_incidents import TRAINING_DATA, CATEGORY_LABELS, SENTIMENT_LABELS
from app.deep_learning.dataset import Vocabulary, Seq2SeqIncidentDataset, PAD_IDX
from app.deep_learning.lstm_classifier import CyberSentinelLSTM
from app.deep_learning.seq2seq_model import IncidentEncoder, IncidentDecoder, IncidentSeq2Seq
from app.deep_learning.evaluator import compute_classification_metrics, compare_baseline_vs_lstm
from app.classifier import CATEGORY_KEYWORDS, classify_incident as baseline_classify_incident
from app.sentiment import classify_sentiment as baseline_classify_sentiment


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def extract_lexical_vector(text: str) -> List[float]:
    """
    Extracts normalized keyword frequency vector across the 7 incident categories.
    """
    text_lower = text.lower()
    vec = []
    for cat in CATEGORY_LABELS:
        keywords = CATEGORY_KEYWORDS.get(cat, [])
        score = sum(1.0 for kw in keywords if kw in text_lower)
        vec.append(score)
    total = sum(vec)
    if total > 0:
        vec = [v / total for v in vec]
    return vec


class FeatureIncidentDataset(Dataset):
    def __init__(self, data, vocab, cat2idx, sent2idx, max_len=32):
        self.samples = []
        for item in data:
            seq = vocab.encode(item["text"], max_len=max_len)
            lex = extract_lexical_vector(item["text"])
            cat_label = cat2idx[item["category"]]
            sent_label = sent2idx[item["sentiment"]]
            self.samples.append((
                torch.tensor(seq, dtype=torch.long),
                torch.tensor(lex, dtype=torch.float32),
                torch.tensor(cat_label, dtype=torch.long),
                torch.tensor(sent_label, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def augment_text(text: str) -> str:
    words = text.split()
    if len(words) > 5 and random.random() < 0.35:
        drop_idx = random.randint(0, len(words) - 1)
        words.pop(drop_idx)
    return " ".join(words)


def train_models():
    print("[INFO] Starting CyberSentinel Deep Learning Training Pipeline...")

    train_data_raw, test_data = train_test_split(
        TRAINING_DATA,
        test_size=0.20,
        random_state=42,
        stratify=[d["category"] for d in TRAINING_DATA]
    )

    augmented_train_data = []
    for item in train_data_raw:
        augmented_train_data.append(item)
        augmented_train_data.append({
            "text": augment_text(item["text"]),
            "category": item["category"],
            "sentiment": item["sentiment"],
            "remediation_action": item["remediation_action"]
        })
        augmented_train_data.append({
            "text": augment_text(item["text"]),
            "category": item["category"],
            "sentiment": item["sentiment"],
            "remediation_action": item["remediation_action"]
        })

    train_data = augmented_train_data
    print(f"[DATA] Base train: {len(train_data_raw)} | Augmented train: {len(train_data)} | Held-out test: {len(test_data)}")

    src_vocab = Vocabulary(min_freq=1)
    src_vocab.build_vocab([d["text"] for d in TRAINING_DATA])

    tgt_vocab = Vocabulary(min_freq=1)
    tgt_vocab.build_vocab([d["remediation_action"] for d in TRAINING_DATA])

    cat2idx = {cat: i for i, cat in enumerate(CATEGORY_LABELS)}
    idx2cat = {i: cat for i, cat in enumerate(CATEGORY_LABELS)}

    sent2idx = {sent: i for i, sent in enumerate(SENTIMENT_LABELS)}
    idx2sent = {i: sent for i, sent in enumerate(SENTIMENT_LABELS)}

    train_cls_ds = FeatureIncidentDataset(train_data, src_vocab, cat2idx, sent2idx)
    test_cls_ds = FeatureIncidentDataset(test_data, src_vocab, cat2idx, sent2idx)

    train_cls_loader = DataLoader(train_cls_ds, batch_size=16, shuffle=True)
    train_s2s_ds = Seq2SeqIncidentDataset(train_data, src_vocab, tgt_vocab)
    train_s2s_loader = DataLoader(train_s2s_ds, batch_size=16, shuffle=True)

    # ----------------------------------------------------
    # 4. Train BiLSTM Classifier
    # ----------------------------------------------------
    print("\n[TRAIN] Training BiLSTM Incident & Sentiment Classifier...")
    lstm_model = CyberSentinelLSTM(
        vocab_size=len(src_vocab),
        embedding_dim=48,
        hidden_dim=64,
        num_categories=len(CATEGORY_LABELS),
        num_sentiments=len(SENTIMENT_LABELS),
        num_layers=1,
        lexical_dim=len(CATEGORY_LABELS),
        bidirectional=True,
        dropout=0.2,
        padding_idx=PAD_IDX
    )

    criterion_cat = nn.CrossEntropyLoss()
    criterion_sent = nn.CrossEntropyLoss()
    optimizer_lstm = optim.AdamW(lstm_model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler_lstm = optim.lr_scheduler.CosineAnnealingLR(optimizer_lstm, T_max=45, eta_min=0.0005)

    epochs_lstm = 45
    lstm_loss_history = []
    lstm_acc_history = []

    for epoch in range(1, epochs_lstm + 1):
        lstm_model.train()
        epoch_loss = 0.0
        correct_cat = 0
        total = 0

        for seqs, lex, cat_labels, sent_labels in train_cls_loader:
            optimizer_lstm.zero_grad()
            cat_logits, sent_logits = lstm_model(seqs, lex)

            loss_c = criterion_cat(cat_logits, cat_labels)
            loss_s = criterion_sent(sent_logits, sent_labels)
            loss = loss_c + 0.8 * loss_s

            loss.backward()
            optimizer_lstm.step()

            epoch_loss += loss.item()
            pred_cats = cat_logits.argmax(dim=1)
            correct_cat += (pred_cats == cat_labels).sum().item()
            total += cat_labels.size(0)

        scheduler_lstm.step()
        avg_loss = epoch_loss / len(train_cls_loader)
        train_acc = correct_cat / total
        lstm_loss_history.append(round(avg_loss, 4))
        lstm_acc_history.append(round(train_acc, 4))

        if epoch % 15 == 0 or epoch == epochs_lstm:
            print(f"  [LSTM Epoch {epoch:02d}/{epochs_lstm}] Loss: {avg_loss:.4f} | Cat Acc: {train_acc*100:.1f}%")

    # ----------------------------------------------------
    # 5. Train Seq2Seq Encoder-Decoder
    # ----------------------------------------------------
    print("\n[TRAIN] Training Encoder-Decoder (Seq2Seq) Model...")
    encoder = IncidentEncoder(input_dim=len(src_vocab), emb_dim=48, hidden_dim=64, n_layers=1, dropout=0.2)
    decoder = IncidentDecoder(output_dim=len(tgt_vocab), emb_dim=48, hidden_dim=64, n_layers=1, dropout=0.2)
    seq2seq_model = IncidentSeq2Seq(encoder, decoder)

    criterion_s2s = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer_s2s = optim.Adam(seq2seq_model.parameters(), lr=0.005)
    scheduler_s2s = optim.lr_scheduler.CosineAnnealingLR(optimizer_s2s, T_max=40, eta_min=0.0005)

    epochs_s2s = 40
    s2s_loss_history = []

    for epoch in range(1, epochs_s2s + 1):
        seq2seq_model.train()
        epoch_loss = 0.0

        for src, trg in train_s2s_loader:
            optimizer_s2s.zero_grad()
            output = seq2seq_model(src, trg, teacher_forcing_ratio=0.5)

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion_s2s(output, trg)
            loss.backward()
            optimizer_s2s.step()

            epoch_loss += loss.item()

        scheduler_s2s.step()
        avg_loss = epoch_loss / len(train_s2s_loader)
        s2s_loss_history.append(round(avg_loss, 4))

        if epoch % 15 == 0 or epoch == epochs_s2s:
            print(f"  [Seq2Seq Epoch {epoch:02d}/{epochs_s2s}] Loss: {avg_loss:.4f}")

    # ----------------------------------------------------
    # 6. Evaluate on Held-Out Test Set
    # ----------------------------------------------------
    print("\n[EVAL] Evaluating Models on Held-Out Test Set...")
    lstm_model.eval()

    test_seqs = torch.stack([s[0] for s in test_cls_ds])
    test_lex = torch.stack([s[1] for s in test_cls_ds])
    test_true_cat_idx = [s[2].item() for s in test_cls_ds]
    test_true_sent_idx = [s[3].item() for s in test_cls_ds]

    with torch.no_grad():
        cat_logits, sent_logits = lstm_model(test_seqs, test_lex)
        lstm_cat_preds_idx = cat_logits.argmax(dim=1).tolist()
        lstm_sent_preds_idx = sent_logits.argmax(dim=1).tolist()

    cat_metrics = compute_classification_metrics(test_true_cat_idx, lstm_cat_preds_idx, CATEGORY_LABELS)
    sent_metrics = compute_classification_metrics(test_true_sent_idx, lstm_sent_preds_idx, SENTIMENT_LABELS)

    # Baseline comparison
    baseline_cat_preds = []
    baseline_sent_preds = []
    true_cat_strings = [test_data[i]["category"] for i in range(len(test_data))]
    true_sent_strings = [test_data[i]["sentiment"] for i in range(len(test_data))]
    lstm_cat_preds_strings = [idx2cat[i] for i in lstm_cat_preds_idx]
    lstm_sent_preds_strings = [idx2sent[i] for i in lstm_sent_preds_idx]

    for item in test_data:
        b_cat = baseline_classify_incident(item["text"])["category"]
        b_sent = baseline_classify_sentiment(item["text"])["sentiment"]
        baseline_cat_preds.append(b_cat)
        baseline_sent_preds.append(b_sent)

    comparison = compare_baseline_vs_lstm(
        baseline_cat_preds, lstm_cat_preds_strings, true_cat_strings,
        baseline_sent_preds, lstm_sent_preds_strings, true_sent_strings
    )

    print(f"  Category LSTM Accuracy: {cat_metrics['accuracy']*100:.1f}% | Macro F1: {cat_metrics['macro_f1']:.4f}")
    print(f"  Sentiment LSTM Accuracy: {sent_metrics['accuracy']*100:.1f}% | Macro F1: {sent_metrics['macro_f1']:.4f}")
    print(f"  Baseline vs LSTM (Category): Baseline {comparison['category']['baseline_accuracy']*100:.1f}% -> LSTM {comparison['category']['lstm_accuracy']*100:.1f}%")
    print(f"  Baseline vs LSTM (Sentiment): Baseline {comparison['sentiment']['baseline_accuracy']*100:.1f}% -> LSTM {comparison['sentiment']['lstm_accuracy']*100:.1f}%")

    # ----------------------------------------------------
    # 7. Save Weights & Artifacts
    # ----------------------------------------------------
    torch.save(lstm_model.state_dict(), os.path.join(MODELS_DIR, "lstm_classifier.pt"))
    torch.save(seq2seq_model.state_dict(), os.path.join(MODELS_DIR, "seq2seq_remediation.pt"))

    metadata = {
        "src_vocab": src_vocab.word2idx,
        "tgt_vocab": tgt_vocab.word2idx,
        "cat2idx": cat2idx,
        "idx2cat": idx2cat,
        "sent2idx": sent2idx,
        "idx2sent": idx2sent,
        "src_vocab_size": len(src_vocab),
        "tgt_vocab_size": len(tgt_vocab),
        "category_labels": CATEGORY_LABELS,
        "sentiment_labels": SENTIMENT_LABELS,
        "embedding_dim": 48,
        "hidden_dim": 64
    }

    metrics_data = {
        "category_metrics": cat_metrics,
        "sentiment_metrics": sent_metrics,
        "comparison": comparison,
        "test_sample_count": len(test_data),
        "train_sample_count": len(train_data)
    }

    history_data = {
        "lstm_loss": lstm_loss_history,
        "lstm_accuracy": lstm_acc_history,
        "seq2seq_loss": s2s_loss_history,
        "epochs": epochs_lstm
    }

    with open(os.path.join(MODELS_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    with open(os.path.join(MODELS_DIR, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2)

    print("\n[SUCCESS] Training Complete! All model weights and evaluation metrics saved successfully.")
    return metrics_data


if __name__ == "__main__":
    train_models()
