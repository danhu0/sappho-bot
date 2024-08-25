import torch
import torch.nn as nn
import torch.optim as optim

class TextGenerationModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(TextGenerationModel, self).__init__()
        self.lstm = nn.LSTM(vocab_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.hidden = hidden_size

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden),
                torch.zeros(1, batch_size, self.hidden))