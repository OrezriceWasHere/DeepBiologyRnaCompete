import torch.nn as nn
import torch.nn.functional as F
import torch

from hyper_parmas import HyperParams


class PredictionModel(nn.Module):

    def __init__(self, args: HyperParams):
        super().__init__()

        self.lstm_cell = nn.LSTM(input_size=args.one_hot_size,
                                 hidden_size=args.lstm_hidden_size,
                                 num_layers=args.lstm_layers,
                                 bidirectional=args.is_bidirectional,
                                 batch_first=True)
        self.fc1 = nn.Linear(args.lstm_hidden_size, args.prediction_classes)
        self.args: HyperParams = args

    def forward(self, x):
        # res = torch.zeros(x.size(0), x.size(1), dtype=torch.float)
        # seq_len = x.size(2)
        # for idx in range(x.size(0)):
        #     cur_tensor = x[idx, :, :]
        #     cur_tensor = cur_tensor.unsqueeze(0)
        #     # Initialize the hidden state and cell state
        #     h_t = torch.zeros(1, 4, dtype=torch.float).to(cur_tensor.device)
        #     c_t = torch.zeros(1, 4, dtype=torch.float).to(cur_tensor.device)
        #
        #     # Process the input sequence step by step
        #     for t in range(seq_len):
        #         x_t = cur_tensor[:, :, t]
        #         h_t, c_t = self.lstm_cell(x_t, (h_t, c_t))
        #
        #     # Fully connected layer
        #     cur_tensor = self.fc1(h_t)
        #     cur_tensor = F.softmax(cur_tensor, dim=1)
        #     res[idx] = cur_tensor
        x_batch_size = x.shape[0]
        hidden_cells_size = (self.args.lstm_layers, x_batch_size, self.args.lstm_hidden_size)
        h_t = torch.zeros(hidden_cells_size, dtype=torch.float).to(x.device)
        c_t = torch.zeros(hidden_cells_size, dtype=torch.float).to(x.device)
        h, c = self.lstm_cell(x, (h_t, c_t))
        last_hidden = h[:, -1, :]
        res = self.fc1(last_hidden)
        res = F.softmax(res, dim=-1)
        return res
