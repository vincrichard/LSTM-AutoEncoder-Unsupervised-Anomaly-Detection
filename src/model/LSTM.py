import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTMVanilla(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_length, batch_size, dropout=0, device=torch.device('cpu')):
        super(LSTMVanilla, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_length = pred_length
        self.batch_size = batch_size
        self.device = device
        self.hidden_cell = (torch.randn((num_layers, batch_size, hidden_size), dtype=torch.float).to(device),
                            torch.randn((num_layers, batch_size, hidden_size), dtype=torch.float).to(device))

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout, bias=True)

        self.linear = nn.Linear(in_features=hidden_size, out_features=pred_length)

    def forward(self, input_seq):
        _, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        predictions = self.linear(self.hidden_cell[0][-1].view(-1, self.hidden_size))
        return predictions

    def detach_hidden(self):
        self.hidden_cell = (self.hidden_cell[0].detach(), self.hidden_cell[1].detach())

    def reset_hidden(self):
        self.hidden_cell = (torch.randn((self.num_layers, self.batch_size, self.hidden_size), dtype=torch.float).to(self.device),
                            torch.randn((self.num_layers, self.batch_size, self.hidden_size), dtype=torch.float).to(self.device))

