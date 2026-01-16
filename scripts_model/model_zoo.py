import tensorflow as tf


# ✅ 모델별로 SEQ_LEN을 여기서 "고정/디폴트"로 관리
MODEL_SPECS = {
    # temporal models
    "lstm":        {"seq_len": 5, "stride": 1},
    "gru":         {"seq_len": 5, "stride": 1},
    "tcn":         {"seq_len": 5, "stride": 1},
    "transformer": {"seq_len": 5, "stride": 1},
    "kalmannet":   {"seq_len": 5, "stride": 1},  # 필요하면 1로 바꿔도 됨

    # single-step models
    "mlp":       {"seq_len": 1, "stride": 1},
    "cnn":       {"seq_len": 1, "stride": 1},
    "convmixer": {"seq_len": 1, "stride": 1},
    "resnet":    {"seq_len": 1, "stride": 1},
}


def get_model_spec(model_name: str):
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_SPECS[model_name]


# ---------------------------
# Model builders (Keras)
# input_shape: (SEQ_LEN, F)
# output: (2,) -> fx, fy
# ---------------------------

def build_mlp(input_shape):
    x = tf.keras.Input(shape=input_shape)
    h = tf.keras.layers.Flatten()(x)  # seq_len=1이면 그냥 (F)랑 동일
    h = tf.keras.layers.Dense(256, activation="relu")(h)
    h = tf.keras.layers.Dense(128, activation="relu")(h)
    h = tf.keras.layers.Dense(64, activation="relu")(h)
    y = tf.keras.layers.Dense(2)(h)
    return tf.keras.Model(x, y, name="MLP")


def build_cnn(input_shape):
    x = tf.keras.Input(shape=input_shape)
    # seq_len=1이면 conv 의미 없으니 Flatten으로 처리
    if input_shape[0] <= 1:
        h = tf.keras.layers.Flatten()(x)
        h = tf.keras.layers.Dense(128, activation="relu")(h)
        y = tf.keras.layers.Dense(2)(h)
        return tf.keras.Model(x, y, name="CNN")

    h = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    h = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(h)
    h = tf.keras.layers.GlobalAveragePooling1D()(h)
    y = tf.keras.layers.Dense(2)(h)
    return tf.keras.Model(x, y, name="CNN")


def build_tcn(input_shape):
    x = tf.keras.Input(shape=input_shape)
    h = tf.keras.layers.Conv1D(64, 3, padding="causal", dilation_rate=1, activation="relu")(x)
    h = tf.keras.layers.Conv1D(64, 3, padding="causal", dilation_rate=2, activation="relu")(h)
    h = tf.keras.layers.Conv1D(64, 3, padding="causal", dilation_rate=4, activation="relu")(h)
    h = tf.keras.layers.GlobalAveragePooling1D()(h)
    y = tf.keras.layers.Dense(2)(h)
    return tf.keras.Model(x, y, name="TCN")


def build_lstm(input_shape):
    x = tf.keras.Input(shape=input_shape)
    h = tf.keras.layers.LSTM(128, return_sequences=True)(x)
    h = tf.keras.layers.LSTM(64)(h)
    h = tf.keras.layers.Dense(32, activation="relu")(h)
    y = tf.keras.layers.Dense(2)(h)
    return tf.keras.Model(x, y, name="LSTM")


def build_gru(input_shape):
    x = tf.keras.Input(shape=input_shape)
    h = tf.keras.layers.GRU(128, return_sequences=True)(x)
    h = tf.keras.layers.GRU(64)(h)
    y = tf.keras.layers.Dense(2)(h)
    return tf.keras.Model(x, y, name="GRU")


def build_transformer(input_shape):
    x = tf.keras.Input(shape=input_shape)
    d_model = 64
    h = tf.keras.layers.Dense(d_model)(x)

    attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=d_model)(h, h)
    h = tf.keras.layers.Add()([h, attn])
    h = tf.keras.layers.LayerNormalization()(h)

    ff = tf.keras.layers.Dense(128, activation="relu")(h)
    ff = tf.keras.layers.Dense(d_model)(ff)
    h = tf.keras.layers.Add()([h, ff])
    h = tf.keras.layers.LayerNormalization()(h)

    h = tf.keras.layers.GlobalAveragePooling1D()(h)
    y = tf.keras.layers.Dense(2)(h)
    return tf.keras.Model(x, y, name="Transformer")


def build_convmixer(input_shape):
    # 가벼운 대체 모델: 1D ConvMixer (seq_len=1이면 MLP처럼 동작)
    x = tf.keras.Input(shape=input_shape)
    if input_shape[0] <= 1:
        h = tf.keras.layers.Flatten()(x)
        h = tf.keras.layers.Dense(128, activation="relu")(h)
        y = tf.keras.layers.Dense(2)(h)
        return tf.keras.Model(x, y, name="ConvMixer")

    h = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    h = tf.keras.layers.DepthwiseConv1D(3, padding="same", activation="relu")(h)
    h = tf.keras.layers.Conv1D(64, 1, padding="same", activation="relu")(h)
    h = tf.keras.layers.GlobalAveragePooling1D()(h)
    y = tf.keras.layers.Dense(2)(h)
    return tf.keras.Model(x, y, name="ConvMixer")


def build_resnet(input_shape):
    # 가벼운 1D ResNet block
    x = tf.keras.Input(shape=input_shape)

    if input_shape[0] <= 1:
        h = tf.keras.layers.Flatten()(x)
        h = tf.keras.layers.Dense(128, activation="relu")(h)
        y = tf.keras.layers.Dense(2)(h)
        return tf.keras.Model(x, y, name="ResNet1D")

    def block(t, filters):
        shortcut = t
        t = tf.keras.layers.Conv1D(filters, 3, padding="same", activation="relu")(t)
        t = tf.keras.layers.Conv1D(filters, 3, padding="same")(t)
        if shortcut.shape[-1] != filters:
            shortcut = tf.keras.layers.Conv1D(filters, 1, padding="same")(shortcut)
        t = tf.keras.layers.Add()([t, shortcut])
        t = tf.keras.layers.Activation("relu")(t)
        return t

    h = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    h = block(h, 64)
    h = block(h, 64)
    h = tf.keras.layers.GlobalAveragePooling1D()(h)
    y = tf.keras.layers.Dense(2)(h)
    return tf.keras.Model(x, y, name="ResNet1D")


def build_kalmannet_lite(input_shape):
    # "KalmanNet" 풀 버전은 별도 설계가 필요하고,
    # 여기서는 lightweight gain-like estimator 형태로 둔다.
    # (SEQ_LEN, F) -> hidden -> (2)
    x = tf.keras.Input(shape=input_shape)
    h = tf.keras.layers.GRU(64)(x)  # temporal encoder
    h = tf.keras.layers.Dense(64, activation="relu")(h)
    y = tf.keras.layers.Dense(2)(h)
    return tf.keras.Model(x, y, name="KalmanNetLite")


def build_model(model_name: str, input_shape):
    model_name = model_name.lower()
    if model_name == "mlp": return build_mlp(input_shape)
    if model_name == "cnn": return build_cnn(input_shape)
    if model_name == "tcn": return build_tcn(input_shape)
    if model_name == "lstm": return build_lstm(input_shape)
    if model_name == "gru": return build_gru(input_shape)
    if model_name == "transformer": return build_transformer(input_shape)
    if model_name == "convmixer": return build_convmixer(input_shape)
    if model_name == "resnet": return build_resnet(input_shape)
    if model_name == "kalmannet": return build_kalmannet_lite(input_shape)
    raise ValueError(f"Unknown model: {model_name}")
