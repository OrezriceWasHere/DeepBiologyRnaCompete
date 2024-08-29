import torch.nn as nn
import torch
from torch.nn import functional as F
from hyper_parmas import HyperParams  # Import the HyperParams class for handling model hyperparameters.


# Define the PredictionModel class, which extends the nn.Module base class from PyTorch.
class PredictionModel(nn.Module):

    def __init__(self, args: HyperParams):
        super().__init__()  # Call the parent class's constructor.

        # Define an embedding layer to transform input sequences into dense vectors.
        self.embedding = nn.Embedding(args.embedding_dict_size, args.embedding_vector_size)

        # Define a dropout layer to prevent overfitting before the LSTM layer.
        self.pre_lstm_dropout = nn.Dropout(args.dropout)

        # Define the LSTM cell with configurable input size, hidden size, number of layers, and bidirectionality.
        self.lstm_cell = nn.LSTM(
            input_size=args.one_hot_size,
            hidden_size=args.lstm_hidden_size,
            num_layers=args.lstm_layers,
            bidirectional=args.is_bidirectional,
            batch_first=True
        )

        # Define a sequential network to process the output from the LSTM layer and produce predictions.
        self.post_lstm_net = nn.Sequential(
            nn.LeakyReLU(),  # Activation function for non-linearity.
            nn.Dropout(args.dropout),  # Dropout layer to prevent overfitting.
            nn.Linear(args.lstm_hidden_size, 3 * args.lstm_hidden_size),  # Fully connected layer.
            nn.LeakyReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(3 * args.lstm_hidden_size, args.lstm_hidden_size),  # Fully connected layer.
            nn.LeakyReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.lstm_hidden_size, args.prediction_classes),  # Final output layer.
        )

        # Store the hyperparameters for later use.
        self.args: HyperParams = args

    def forward(self, x, x_size=None):
        # Forward pass through the embedding layer.
        embedded_x = self.embedding(x)

        # Get the batch size from the input tensor.
        x_batch_size = x.shape[0]

        # Initialize hidden and cell states for the LSTM.
        hidden_cells_size = (self.args.lstm_layers, x_batch_size, self.args.lstm_hidden_size)
        h_t = torch.zeros(hidden_cells_size, dtype=torch.float).to(x.device)
        c_t = torch.zeros(hidden_cells_size, dtype=torch.float).to(x.device)

        # Pass the embedded input through the LSTM.
        h, c = self.lstm_cell(embedded_x, (h_t, c_t))

        # Gather the output from the last LSTM cell.
        last_indices = (x_size - 1).unsqueeze(2).expand(x_batch_size, 1, self.args.lstm_hidden_size)
        last_hidden = h.gather(1, last_indices).squeeze(1)

        # Pass the output from the last LSTM cell through the post-LSTM network to get predictions.
        return self.post_lstm_net(last_hidden).squeeze(1)
