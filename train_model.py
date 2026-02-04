import os
import torch
import torch.nn as nn
from model import NextWordRNN
from data_utils import tokenize, vocab_builder, sequential_word_builder

SEQ_LEN = 5
EPOCHS = 10
LR = 0.001

def load_books(path="data"):
    texts = []
    for fname in os.listdir(path):
        if fname.endswith(".txt"):
            with open(os.path.join(path, fname), encoding="utf-8") as f:
                texts.append(f.read())
    return "\n".join(texts)

print("Loading data...")
text = load_books()
words = tokenize(text)
print(f"Total words: {len(words)}")

print("Building Vocab...")
word_to_idx, idx_to_word = vocab_builder(words)

print("Building sequences...")
X, y = sequential_word_builder(words, word_to_idx, SEQ_LEN)
print(f"Database size: {X.shape[0]}")
print(f"Vocab Size: {len(word_to_idx)}")