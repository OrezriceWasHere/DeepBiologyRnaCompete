class HyperParams:
    lr: float = 2.8e-05
    one_hot_size: int = 6
    embedding_vector_size: int = 6
    batch_size: int = 2048
    epochs: int = 60
    lstm_layers: int = 2
    lstm_hidden_size: int = 200
    prediction_classes: int = 4
    embedding_char_length: int = 5
    embedding_dict_size: int = 200
    padding_max_size: int = 20
    is_bidirectional: bool = False


def optuna_suggest_hyperparams(trial) -> HyperParams:
    hyper_params = HyperParams()
    hyper_params.lr = trial.suggest_float('lr', 1e-5, 1e-4, log=True)
    hyper_params.lstm_hidden_size = trial.suggest_int('lstm_hidden_size', 128, 256)
    hyper_params.lstm_layers = trial.suggest_int('lstm_layers', 1, 3)
    hyper_params.embedding_char_length = trial.suggest_int('embedding_length', 3, 5)
    hyper_params.embedding_vector_size = trial.suggest_int('embedding_vector_size', 2, 8)

    return hyper_params

