import os
import cv2
from tqdm import tqdm
import json


def load_config(self, config_file='config_ROI_ref.json'):
    print(f'config.json PATH: {config_file}')

    print(f'Load {config_file}')
    print(f'===== json list ======')
    with open(config_file, 'r') as f:
        config = json.load(f)
        # print
        for key, value in config.items():
            print(f'{key} : {value}')
    print(f'===== json list end ===')

    return config

def process(dir_path):
    config_file = 'config_ROI_ref.json'
    config = load_config(config_file)

    # set ROI
    roi_x, roi_y, roi_w, roi_h = config.get('x'), config.get('y'), config.get('w'), config.get('h')

    # 이미지가 저장된 폴더 경로
    input_dir = os.path.join(dir_path, 'images')
    output_dir = os.path.join(dir_path, 'images_ROI')

    print(f'Processing in dir:{input_dir}')

    # output 디렉토리가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 이미지 파일들을 처리
    list = os.listdir(input_dir)
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # 이미지 파일 포맷에 맞게 확장자 설정
            img_path = os.path.join(input_dir, filename)

            # 이미지 읽기
            img = cv2.imread(img_path)

            if img is not None:
                # ROI 설정 (y:y+h, x:x+w)
                roi = img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

                # 새로운 이미지 파일 저장 경로
                output_path = os.path.join(output_dir, filename)

                # ROI 이미지 저장
                cv2.imwrite(output_path, roi)
                # print(f'{filename}의 ROI가 {output_path}에 저장되었습니다.')
            else:
                print(f'{filename}을(를) 읽을 수 없습니다.')

    print(f'Processing Finish in dir:{output_dir}')


if __name__ == '__main__':
    # define path of directory
    base_dir = '../data/2024-10-10 experiment (0.35 mm) test'

    # 이미지 프로세싱 클래스
    # 최상위 폴더 내의 모든 하위 폴더를 탐색
    subfolders = [os.path.join(base_dir, name) for name in os.listdir(base_dir)
                  if os.path.isdir(os.path.join(base_dir, name))]

    for dir_path in subfolders:
        # 폴더 처리 및 결과 저장
        process(dir_path)

print(f'All ROI images are saved')
