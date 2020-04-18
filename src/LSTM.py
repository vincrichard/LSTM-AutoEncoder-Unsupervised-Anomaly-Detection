import torch
import torch.nn as nn


class LSTMVanilla(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_length, batch_size, dropout=0):
        super(LSTMVanilla, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_length = pred_length
        self.batch_size = batch_size
        self.hidden_cell = (torch.zeros(num_layers, batch_size, hidden_size),
                            torch.zeros(num_layers, batch_size, hidden_size))

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout, bias=True)

        self.linear = nn.Linear(in_features=hidden_size, out_features=pred_length)

    def forward(self, input_seq):
        _, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.linear(self.hidden_cell[0].view(-1, self.hidden_size))
        return predictions

    def reset_memory(self):
        self.hidden_cell = (torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                           torch.zeros(self.num_layers, self.batch_size, self.hidden_size))

