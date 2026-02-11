import torch
import torch.nn.functional as F
from model import NextWordRNN
from data_utils import tokenize

SEQ_LEN = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load("model.pth", map_location=DEVICE)
word_to_idx = checkpoint["word_to_idx"]
idx_to_word = checkpoint["idx_to_word"]

model = NextWordRNN(len(word_to_idx)).to(DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

def suggest_next(text, k=5, temperature=1.0):
    tokens = tokenize(text)[-SEQ_LEN:]
    if len(tokens) < SEQ_LEN:
        return ["<not enough context>"]

    try:
        x = torch.tensor([[word_to_idx[w] for w in tokens]]).to(DEVICE)
    except KeyError as e:
        return [f"<unknown token: {e.args[0]}>"]

    logits = model(x) / temperature
    probs = F.softmax(logits, dim=1)
    topk = torch.topk(probs, k)
    return [idx_to_word[i.item()] for i in topk.indices[0]]

print("Type some words and press enter to see suggested words (or 'quit' to exit):")
while True:
    prompt = input("> ")
    if prompt.lower() == "quit":
        break
    suggestions = suggest_next(prompt)
    print("Suggestions:", suggestions)