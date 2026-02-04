import torch.nn as nn

class NextWordRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, dropout_p=0.1):
        super(NextWordRNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input):
        embedded = self.embedding(self.dropout(input))
        output, hidden = self.gru(embedded)
        
        last_hidden = output[:, -1, :]
        out = self.fc(last_hidden)
        return out