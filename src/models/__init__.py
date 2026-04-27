from .architectures import CNN1D, BiLSTM, BiGRU, MiniBERT

def get_model(model_name, num_classes=12):
    if model_name == "1d_cnn":
        return CNN1D(num_classes=num_classes)
    elif model_name == "bi_lstm":
        return BiLSTM(num_classes=num_classes)
    elif model_name == "bi_gru":
        return BiGRU(num_classes=num_classes)
    elif model_name == "minibert":
        return MiniBERT(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
