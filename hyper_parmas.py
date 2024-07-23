class HyperParams:
    lr: float = 1e-3
    one_hot_size: int = 5
    batch_size: int = 4096
    epochs: int = 20
    lstm_layers: int = 1
    lstm_hidden_size: int = 50
    is_bidirectional: bool = False
    prediction_classes: int = 4
    # kernel_size: int = 3
    # conv_out_channels: int = 4
