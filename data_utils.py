import torch
import re

def tokenize(str):
    str = str.lower().strip()
    str = re.sub(r"([.!?])", r" \1", str)
    str = re.sub(r"[^a-z0-9'.!?]+", " ", str)
    return str.split()

def vocab_builder(words):
    vocab = sorted(set(words))
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}

    return word_to_idx, idx_to_word

def sequential_word_builder(words, word_to_idx, seq_len):
    X, y = [], []
    for i in range(len(words) - seq_len):
        X.append([word_to_idx[w] for w in words[i:i+seq_len]])
        y.append(word_to_idx[words[i+seq_len]])
    return torch.tensor(X), torch.tensor(y)