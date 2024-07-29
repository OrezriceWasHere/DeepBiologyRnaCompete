class HyperParams:
    lr: float = 1e-3
    one_hot_size: int = 6
    batch_size: int = 2048
    epochs: int = 20
    lstm_layers: int = 1
    lstm_hidden_size: int = 128
    prediction_classes: int = 4
    sum_weights_lambda: float = 0.05
    is_bidirectional: bool = False


def optuna_suggest_hyperparams(trial) -> HyperParams:
    hyper_params = HyperParams()
    hyper_params.lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    hyper_params.epochs = trial.suggest_int('epochs', 15, 100)
    hyper_params.lstm_hidden_size = trial.suggest_int('lstm_hidden_size', 32, 512)
    hyper_params.lstm_layers = trial.suggest_int('lstm_layers', 1, 3)
    hyper_params.sum_weights_lambda = trial.suggest_float('sum_weights_lambda', 0.01, 0.1)
    # hyper_params.is_bidirectional = trial.suggest_categorical('is_bidirectional', [True, False])
    return hyper_params

