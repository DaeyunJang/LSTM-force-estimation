from tensorflow.keras.models import load_model

# 모델 로드
model1 = load_model('lstm_model-present.h5')
model2 = load_model('lstm_model-legacy.h5')

# 모델 구조 비교
def compare_model_structures(model1, model2):
    structure1 = [layer.name for layer in model1.layers]
    structure2 = [layer.name for layer in model2.layers]

    if structure1 == structure2:
        print("모델 구조가 동일합니다.")
    else:
        print("모델 구조가 다릅니다.")
        print("Model 1 layers:", structure1)
        print("Model 2 layers:", structure2)

compare_model_structures(model1, model2)

import numpy as np

# 모델 가중치 비교
def compare_weights(model1, model2):
    weights1 = model1.get_weights()
    weights2 = model2.get_weights()

    if len(weights1) != len(weights2):
        print("가중치의 개수가 다릅니다.")
        return False

    for i, (w1, w2) in enumerate(zip(weights1, weights2)):
        if not np.array_equal(w1, w2):
            print(f"가중치가 다른 레이어: {i}")
            return False

    print("모든 가중치가 동일합니다.")
    return True

compare_weights(model1, model2)


# 컴파일 설정 비교
def compare_compile_settings(model1, model2):
    config1 = model1.optimizer.get_config() if model1.optimizer else None
    config2 = model2.optimizer.get_config() if model2.optimizer else None

    if config1 == config2:
        print("컴파일 설정이 동일합니다.")
    else:
        print("컴파일 설정이 다릅니다.")
        print("Model 1 optimizer config:", config1)
        print("Model 2 optimizer config:", config2)

compare_compile_settings(model1, model2)
