import math  # Import the math library for mathematical operations.


# Define a class to hold hyperparameters for the model.
class HyperParams:
    # Learning rate for the optimizer.
    lr: float = 6.12e-6
    # Size of one-hot encoded vectors.
    one_hot_size: int = 11
    # Size of the embedding vectors.
    embedding_vector_size: int = 11
    # Batch size used during training.
    batch_size: int = 128
    # Number of epochs for training.
    epochs: int = 35
    # Number of layers in the LSTM network.
    lstm_layers: int = 1
    # Number of hidden units in the LSTM layers.
    lstm_hidden_size: int = 221
    # Number of prediction classes (output size).
    prediction_classes: int = 4
    # Length of characters used in embedding.
    embedding_char_length: int = 4
    # Size of the dictionary used for embedding (vocabulary size).
    embedding_dict_size: int = 1228
    # Maximum padding size to ensure uniform input length.
    padding_max_size: int = 20
    # Flag to indicate if the LSTM is bidirectional.
    is_bidirectional: bool = False
    # Dropout rate for regularization to prevent overfitting.
    dropout: float = 0.4


# Function to suggest hyperparameters using the Optuna optimization framework.
def optuna_suggest_hyperparams(trial) -> HyperParams:
    # Create an instance of the HyperParams class.
    hp = HyperParams()

    # Suggest a learning rate between 1e-6 and 1e-4 on a logarithmic scale.
    hp.lr = trial.suggest_float('lr', 1e-6, 1e-4, log=True)

    # Suggest a hidden size for the LSTM layers between 128 and 256 units.
    hp.lstm_hidden_size = trial.suggest_int('lstm_hidden_size', 128, 256)

    # Suggest the number of LSTM layers, choosing between 1 and 3.
    hp.lstm_layers = trial.suggest_int('lstm_layers', 1, 3)

    # Suggest the character length used in the embedding, between 3 and 7.
    hp.embedding_char_length = trial.suggest_int('embedding_length', 3, 7)

    # Suggest the size of embedding vectors, between 2 and 20 dimensions.
    hp.embedding_vector_size = trial.suggest_int('embedding_vector_size', 2, 20)

    # Suggest a dropout rate from a predefined set of values.
    hp.dropout = trial.suggest_categorical('dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5])

    # Set the one-hot size to match the embedding vector size.
    hp.one_hot_size = hp.embedding_vector_size

    # Calculate the dictionary size for the embeddings based on the character length.
    # The size is scaled by a factor of 1.2 and adjusted using the math.floor function.
    hp.embedding_dict_size = math.floor((4 ** (hp.embedding_char_length + 1)) * 1.2)

    return hp
