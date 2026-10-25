import re
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple


# Special Tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


def simple_tokenize(text: str) -> List[str]:
    """
    NLP Tokenization: Cleans and splits text into individual word tokens.
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t]
    return tokens


class Vocabulary:
    """
    Vocabulary builder for numerical representation of tokens.
    Maps words to unique integers and vice versa.
    """
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {
            PAD_TOKEN: PAD_IDX,
            UNK_TOKEN: UNK_IDX,
            SOS_TOKEN: SOS_IDX,
            EOS_TOKEN: EOS_IDX,
        }
        self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}
        self.word_counts: Dict[str, int] = {}

    def build_vocab(self, texts: List[str]):
        for text in texts:
            tokens = simple_tokenize(text)
            for token in tokens:
                self.word_counts[token] = self.word_counts.get(token, 0) + 1

        idx = len(self.word2idx)
        for token, count in sorted(self.word_counts.items(), key=lambda x: -x[1]):
            if count >= self.min_freq and token not in self.word2idx:
                self.word2idx[token] = idx
                self.idx2word[idx] = token
                idx += 1

    def encode(self, text: str, max_len: int = 32, add_sos_eos: bool = False) -> List[int]:
        tokens = simple_tokenize(text)
        indices = []
        if add_sos_eos:
            indices.append(SOS_IDX)

        for token in tokens:
            indices.append(self.word2idx.get(token, UNK_IDX))

        if add_sos_eos:
            indices.append(EOS_IDX)

        # Truncate or Pad
        if len(indices) < max_len:
            indices += [PAD_IDX] * (max_len - len(indices))
        else:
            indices = indices[:max_len]
            if add_sos_eos:
                indices[-1] = EOS_IDX

        return indices

    def decode(self, indices: List[int], stop_at_eos: bool = True) -> str:
        words = []
        for idx in indices:
            if idx in (PAD_IDX, SOS_IDX):
                continue
            if stop_at_eos and idx == EOS_IDX:
                break
            words.append(self.idx2word.get(idx, UNK_TOKEN))
        return " ".join(words)

    def __len__(self):
        return len(self.word2idx)


class IncidentClassificationDataset(Dataset):
    """
    PyTorch Dataset for Incident Category & Sentiment Classification.
    """
    def __init__(self, data: List[Dict], vocab: Vocabulary, cat2idx: Dict[str, int], sent2idx: Dict[str, int], max_len: int = 32):
        self.samples = []
        for item in data:
            seq = vocab.encode(item["text"], max_len=max_len)
            cat_label = cat2idx[item["category"]]
            sent_label = sent2idx[item["sentiment"]]
            self.samples.append((
                torch.tensor(seq, dtype=torch.long),
                torch.tensor(cat_label, dtype=torch.long),
                torch.tensor(sent_label, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class Seq2SeqIncidentDataset(Dataset):
    """
    PyTorch Dataset for Incident Description -> Remediation Playbook (Encoder-Decoder Seq2Seq).
    """
    def __init__(self, data: List[Dict], src_vocab: Vocabulary, tgt_vocab: Vocabulary, src_max_len: int = 32, tgt_max_len: int = 24):
        self.samples = []
        for item in data:
            src_seq = src_vocab.encode(item["text"], max_len=src_max_len, add_sos_eos=False)
            tgt_seq = tgt_vocab.encode(item["remediation_action"], max_len=tgt_max_len, add_sos_eos=True)
            self.samples.append((
                torch.tensor(src_seq, dtype=torch.long),
                torch.tensor(tgt_seq, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
