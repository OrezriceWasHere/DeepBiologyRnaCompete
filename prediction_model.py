import torch.nn as nn
import torch

from hyper_parmas import HyperParams


class PredictionModel(nn.Module):

    def __init__(self, args: HyperParams):
        super().__init__()
        self.embedding = nn.Embedding(args.embedding_dict_size, args.embedding_vector_size)
        self.lstm_cell = nn.LSTM(input_size=args.one_hot_size,
                                 hidden_size=args.lstm_hidden_size,
                                 num_layers=args.lstm_layers,
                                 bidirectional=args.is_bidirectional,
                                 batch_first=True)
        self.fc1 = nn.Linear(args.lstm_hidden_size, args.prediction_classes)
        self.args: HyperParams = args

    def forward(self, x, x_size=None):
        embedded_x = self.embedding(x)
        x_batch_size = x.shape[0]
        hidden_cells_size = (self.args.lstm_layers, x_batch_size, self.args.lstm_hidden_size)
        h_t = torch.zeros(hidden_cells_size, dtype=torch.float).to(x.device)
        c_t = torch.zeros(hidden_cells_size, dtype=torch.float).to(x.device)
        h, c = self.lstm_cell(embedded_x, (h_t, c_t))
        last_indices = (x_size - 1).unsqueeze(2).expand(x_batch_size, 1, self.args.lstm_hidden_size)
        last_hidden = h.gather(1, last_indices).squeeze(1)
        res = self.fc1(last_hidden)
        return res
