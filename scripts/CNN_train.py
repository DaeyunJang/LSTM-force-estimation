import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
from glob import glob
import datetime
import os
from sklearn.model_selection import train_test_split

# 모델 저장 디렉토리 설정
save_dir = os.path.join('..', 'fit_CNN')
save_dir = os.path.join(save_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# 1D CNN 모델 정의
def create_1d_cnn_model(input_shape):
    model = tf.keras.models.Sequential()

    # 1D 합성곱 레이어: 패딩을 'same'으로 설정하여 입력과 출력 크기를 동일하게 맞춤
    model.add(
        tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

    # 두 번째 Conv1D 레이어: 필터 개수를 64로 설정, 'same' 패딩 적용
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

    # 세 번째 Conv1D 레이어 추가: 필터 개수를 32로 설정, 'same' 패딩 적용
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

    # Flatten 레이어: 1차원 벡터로 변환
    model.add(tf.keras.layers.Flatten())

    # 완전 연결층(Dense): 유닛 수 100, 활성화 함수 ReLU
    model.add(tf.keras.layers.Dense(100, activation='relu'))

    # 드롭아웃 레이어 추가: 과적합 방지를 위해 20%의 뉴런을 무작위로 드롭
    # model.add(tf.keras.layers.Dropout(0.2))

    # 두 번째 완전 연결층(Dense): 유닛 수 50
    model.add(tf.keras.layers.Dense(50, activation='relu'))

    # 출력층: fx와 fy 두 개의 출력을 위한 2개의 유닛
    model.add(tf.keras.layers.Dense(2))

    # 컴파일: Huber Loss 사용, 옵티마이저는 Adam
    model.compile(optimizer='adam', loss=tf.keras.losses.Huber())
    # model.compile(optimizer='adam', loss='huber_loss')

    return model


# 데이터 전처리
def load_and_preprocess_data():
    # 여러 개의 CSV와 JSON 파일 경로를 지정합니다.
    train_csv = sorted(glob('../datasets/train/data_LPF_2*.csv'))
    curvefit_json = sorted(glob('../datasets/train/curve_fit_result-joint_angle_*.json'))

    # 모든 CSV 파일을 읽어 리스트에 저장합니다.
    csv_dataframes = [pd.read_csv(file) for file in train_csv]

    # 모든 JSON 파일을 읽어 리스트에 저장합니다.
    json_dataframes = [pd.read_json(file) for file in curvefit_json]

    # 병합된 CSV와 JSON 데이터를 하나의 데이터프레임으로 병합합니다.
    raw_dataframe = pd.concat([df for df in csv_dataframes])
    curvefit_dataframe = pd.concat([df for df in json_dataframes])
    data_expanded = pd.concat([raw_dataframe, curvefit_dataframe], axis=1)

    # Curve fitting에서 Joint Angle 배열을 분리
    column_size = len(data_expanded['Joint Angle'].iloc[0])
    joint_angle = np.array(data_expanded['Joint Angle'].tolist())
    joint_angle_df = pd.DataFrame(joint_angle, columns=[f'Joint Angle_{i}' for i in range(column_size)])

    # 기존 데이터프레임과 Joint Angle 개별 열을 합침
    final_df = pd.concat([data_expanded.reset_index(drop=True), joint_angle_df], axis=1)

    # 입력과 출력 데이터 분리
    input_columns = ['wire length #0', 'wire length #1', 'loadcell #0', 'loadcell #1'] + [f'Joint Angle_{i}' for i in range(column_size)]
    output_columns = ['fx', 'fy']

    x = final_df[input_columns].values
    y = final_df[output_columns].values

    # 데이터 정규화
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    x_normalized = scaler_x.fit_transform(x)
    y_normalized = scaler_y.fit_transform(y)

    # 학습/검증 데이터 분할
    x_train, x_valid, y_train, y_valid = train_test_split(x_normalized, y_normalized, test_size=0.2, random_state=42)

    # 입력 데이터 차원 조정 (CNN에 맞도록 차원 확장)
    x_train = np.expand_dims(x_train, axis=-1)  # (샘플 수, 피처 수, 1)
    x_valid = np.expand_dims(x_valid, axis=-1)

    # 스케일러 저장
    joblib.dump(scaler_x, os.path.join(save_dir, 'scaler_x.pkl'))
    joblib.dump(scaler_y, os.path.join(save_dir, 'scaler_y.pkl'))

    return x_train, x_valid, y_train, y_valid


# 데이터 로드 및 전처리
x_train, x_valid, y_train, y_valid = load_and_preprocess_data()

# CNN 모델 생성
input_shape = (x_train.shape[1], 1)  # 피처 수에 맞춘 입력 형상
model = create_1d_cnn_model(input_shape)

# 모델 학습
history = model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_valid, y_valid))

# 학습한 모델 저장
model.save(os.path.join(save_dir, 'cnn_model.h5'))

print("모델이 저장되었습니다.")

# 학습 곡선 그리기
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()


