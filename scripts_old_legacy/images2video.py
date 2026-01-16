import cv2
import os
from natsort import natsorted
import imageio
from tqdm import tqdm

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory {directory_path} created.")
    else:
        print(f"Directory {directory_path} already exists.")

# 이미지가 저장된 디렉토리 경로
name = 'images_with_arrow_20241007_070507'
image_folder = os.path.join('../data/2024-10-01 experiment (0.4 mm)/2024-10-01-16-32-03 Super Random', name)

create_directory('videos')
video_name = 'videos/' + name + '.avi'
video_name_mp4 = 'videos/' + name + '.mp4'
gif_name = 'videos/' + name + '.gif'
fps = 30

# 이미지 파일 이름들을 정렬된 리스트로 가져오기
images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]
images = natsorted(images)  # 자연스러운 숫자 순서로 파일 이름 정렬

# 첫 번째 이미지로 비디오의 프레임 크기를 설정
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# AVI 비디오 작성기 초기화 (초당 프레임 수 30으로 설정)
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
video2 = cv2.VideoWriter(video_name_mp4, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

# 프레임을 저장할 리스트 생성 (GIF용)
gif_frames = []

# 이미지들을 비디오로 변환 및 GIF로 저장
for image in tqdm(images):
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)

    # AVI 비디오에 프레임 추가
    video.write(frame)
    video2.write(frame)

    # GIF 프레임에 추가 (BGR을 RGB로 변환)
    gif_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gif_frames.append(gif_frame)

# AVI 비디오 파일 저장
video.release()
video2.release()

# GIF 파일 저장 (10 FPS로 설정)
imageio.mimsave(gif_name, gif_frames, fps=fps)

print(f"AVI 비디오 '{video_name}'와 GIF 파일 '{gif_name}'가 성공적으로 생성되었습니다.")
