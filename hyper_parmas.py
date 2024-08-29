# HyperParams class defines the hyperparameters used for training the LSTM-based prediction model.
class HyperParams:
    lr: float = 1e-3  # Learning rate for the optimizer.
    one_hot_size: int = 6  # Size of the one-hot encoded vectors for the input data.
    batch_size: int = 2048  # Number of samples processed before the model is updated.
    epochs: int = 20  # Number of complete passes through the training dataset.
    lstm_layers: int = 1  # Number of LSTM layers in the model.
    lstm_hidden_size: int = 128  # Number of features in the hidden state of the LSTM.
    prediction_classes: int = 4  # Number of output classes for the prediction task.
    is_bidirectional: bool = False  # Whether the LSTM is bidirectional or not.


# optuna_suggest_hyperparams function suggests a set of hyperparameters to be used in training.
def optuna_suggest_hyperparams(trial) -> HyperParams:
    hyper_params = HyperParams()
    # Suggests a learning rate in the range of 1e-5 to 1e-1 on a log scale.
    hyper_params.lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    # Suggests the number of epochs between 15 and 100.
    hyper_params.epochs = trial.suggest_int('epochs', 15, 100)
    # Suggests the LSTM hidden size between 32 and 512 units.
    hyper_params.lstm_hidden_size = trial.suggest_int('lstm_hidden_size', 32, 512)
    # Suggests the number of LSTM layers between 1 and 3.
    hyper_params.lstm_layers = trial.suggest_int('lstm_layers', 1, 3)
    # hyper_params.is_bidirectional = trial.suggest_categorical('is_bidirectional', [True, False])
    return hyper_params
