import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from model import NextWordRNN
from data_utils import tokenize, vocab_builder, sequential_word_builder

SEQ_LEN = 5
EPOCHS = 5
LR = 0.0005
BATCH_SIZE = 512
VAL_SPLIT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Text(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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
print("Ready to train\n")

text_dataset = Text(X, y)

val_size = int(len(text_dataset) * VAL_SPLIT)
train_size = len(text_dataset) - val_size

train_data, val_data = random_split(text_dataset, [train_size, val_size])
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)



# Model, training and saving


model = NextWordRNN(len(word_to_idx)).to(DEVICE)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("Training model")
for epoch in range(EPOCHS):
    model.train()

    training_loss = 0
    for batch_X, batch_y in train_dataloader:
        batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss : torch.Tensor = loss_fn(outputs, batch_y)
        loss.backward()
        optimizer.step()
        training_loss += loss.item() * batch_X.size(0)
    
    training_loss /= len(train_dataloader.dataset)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_dataloader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            outputs = model(batch_X)
            loss : torch.Tensor =  loss_fn(outputs, batch_y)
            val_loss += loss.item() * batch_X.size(0)
        val_loss /= len(val_dataloader.dataset)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {training_loss:.4f} | Val Loss: {val_loss:.4f}")

torch.save({
    "model_state": model.state_dict(),
    "word_to_idx": word_to_idx,
    "idx_to_word": idx_to_word
}, "model.pth")
print("Saved Pytorch model as model.pth")