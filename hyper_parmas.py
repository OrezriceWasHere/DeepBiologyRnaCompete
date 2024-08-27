import math

class HyperParams:
    lr: float = 6.12e-5
    one_hot_size: int = 11
    embedding_vector_size: int = 11
    batch_size: int = 128
    epochs: int = 35
    lstm_layers: int = 1
    lstm_hidden_size: int = 221
    prediction_classes: int = 4
    embedding_char_length: int = 4
    embedding_dict_size: int = 1228
    padding_max_size: int = 20
    is_bidirectional: bool = False
    dropout: float = 0.4


def optuna_suggest_hyperparams(trial) -> HyperParams:
    hp = HyperParams()
    hp.lr = trial.suggest_float('lr', 1e-6, 1e-4, log=True)
    hp.lstm_hidden_size = trial.suggest_int('lstm_hidden_size', 128, 256)
    hp.lstm_layers = trial.suggest_int('lstm_layers', 1, 3)
    hp.embedding_char_length = trial.suggest_int('embedding_length', 3, 5)
    hp.embedding_vector_size = trial.suggest_int('embedding_vector_size', 2, 20)
    hp.dropout = trial.suggest_categorical('dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5])
    hp.one_hot_size = hp.embedding_vector_size
    hp.embedding_dict_size = math.floor((4 ** (hp.embedding_char_length + 1)) * 1.2)
    return hp
