import os
import re
import cv2
from pathlib import Path

# ======================
# 설정
# ======================
INPUT_DIR = r"../data/2025-12-31_Fe_train_datasets/2025-12-31_Fe_train_dataset_dynamic_3/images"          # 이미지 폴더 경로
OUTPUT_MP4 = r"./output.mp4"     # 저장할 영상 경로
FPS = 30

# 지원 확장자
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ======================
# 파일명 숫자 기준 정렬
# - 예: "12_abc.png", "img_0003.png" 등 -> 숫자들을 뽑아 tuple로 정렬
# - 숫자가 없는 파일은 뒤로 감
# ======================
num_re = re.compile(r"\d+")

def numeric_key(p: Path):
    nums = [int(x) for x in num_re.findall(p.stem)]
    if nums:
        return (0, nums, p.name.lower())  # 숫자 있는 것 우선
    return (1, [], p.name.lower())        # 숫자 없는 것 나중

# ======================
# 이미지 목록 수집
# ======================
in_dir = Path(INPUT_DIR)
if not in_dir.exists():
    raise FileNotFoundError(f"INPUT_DIR not found: {in_dir}")

images = [p for p in in_dir.iterdir() if p.suffix.lower() in EXTS]
images.sort(key=numeric_key)

if not images:
    raise RuntimeError(f"No images found in: {in_dir}")

print(f"Found {len(images)} images.")
print("First 5:", [p.name for p in images[:5]])

# ======================
# 첫 프레임으로 해상도 결정
# ======================
first = cv2.imread(str(images[0]))
if first is None:
    raise RuntimeError(f"Failed to read first image: {images[0]}")

h, w = first.shape[:2]

# ======================
# VideoWriter 설정 (mp4)
# ======================
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_MP4, fourcc, FPS, (w, h))

if not writer.isOpened():
    raise RuntimeError("Failed to open VideoWriter. Try changing codec or output path.")

# ======================
# 프레임 쓰기
# - 해상도 다른 이미지는 첫 프레임 크기로 강제 리사이즈
# ======================
for i, p in enumerate(images):
    img = cv2.imread(str(p))
    if img is None:
        print(f"[WARN] Skipping unreadable: {p.name}")
        continue

    if img.shape[1] != w or img.shape[0] != h:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    writer.write(img)

    if (i + 1) % 200 == 0:
        print(f"Written {i+1}/{len(images)} frames...")

writer.release()
print(f"Done. Saved: {OUTPUT_MP4}")
