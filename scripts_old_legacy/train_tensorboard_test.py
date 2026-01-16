import tensorflow as tf
import datetime

# TensorBoard 로그 디렉토리 설정
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# 간단한 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 임의의 데이터로 학습
X_train = tf.random.normal((1000, 5))
y_train = tf.random.normal((1000, 1))

# 모델 훈련
model.fit(X_train, y_train, epochs=5, callbacks=[tensorboard_callback])

# TensorBoard 실행 후 로그 확인
