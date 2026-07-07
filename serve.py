import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from model import NextWordRNN
from data_utils import tokenize

SEQ_LEN = 15
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

    with torch.no_grad():
        logits = model(x) / temperature
        probs = F.softmax(logits, dim=1)
        topk = torch.topk(probs, k)

    words = [idx_to_word[i.item()] for i in topk.indices[0]]
    scores = [round(p.item(), 4) for p in topk.values[0]]
    return words, scores


class PredictRequest(BaseModel):
    text: str
    k: int = 5
    temperature: float = 1.0


app = FastAPI(title="Next Word Predictor")


@app.get("/health")
def health():
    return {"status": "ok", "vocab_size": len(word_to_idx), "device": str(DEVICE)}


@app.post("/predict")
def predict(req: PredictRequest):
    result = suggest_next(req.text, k=req.k, temperature=req.temperature)
    if isinstance(result, list):
        return {"input": req.text, "suggestions": [], "message": result[0]}
    words, scores = result
    return {
        "input": req.text,
        "suggestions": [{"word": w, "probability": s} for w, s in zip(words, scores)],
    }
