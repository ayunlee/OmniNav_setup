"""
SocialACT 10hz 데이터를 2hz로 다운샘플링하는 스크립트.
- 원본: SocialACT/ 안의 10개 시퀀스 (10hz, frame_0000 ~ frame_XXXX)
- 출력: SocialACT_2hz/ 에 매 5번째 프레임만 복사, 인덱스를 0부터 순차 재배정
- 각 프레임 폴더 내 파일 이름도 새 인덱스에 맞게 변경됨
"""

import os
import shutil
import re

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "SocialACT")
DST_DIR = os.path.join(BASE_DIR, "SocialACT_2hz")

SAMPLE_STEP = 5  # 10hz -> 2hz: 5프레임마다 1개 선택


def get_sorted_frame_dirs(seq_path):
    """frame_XXXX 형식의 디렉토리 목록을 숫자 순으로 정렬하여 반환."""
    frame_dirs = []
    for d in os.listdir(seq_path):
        if os.path.isdir(os.path.join(seq_path, d)) and re.match(r"frame_\d+", d):
            frame_dirs.append(d)
    frame_dirs.sort(key=lambda x: int(re.search(r"\d+", x).group()))
    return frame_dirs


def downsample_sequence(seq_name):
    """단일 시퀀스에 대해 2hz 다운샘플링 수행."""
    src_seq = os.path.join(SRC_DIR, seq_name)
    dst_seq = os.path.join(DST_DIR, seq_name)

    frame_dirs = get_sorted_frame_dirs(src_seq)
    total_frames = len(frame_dirs)

    # 매 SAMPLE_STEP번째 프레임 선택 (0, 5, 10, ...)
    selected = frame_dirs[::SAMPLE_STEP]

    os.makedirs(dst_seq, exist_ok=True)

    for new_idx, old_frame_dir in enumerate(selected):
        old_frame_path = os.path.join(src_seq, old_frame_dir)
        new_frame_name = f"frame_{new_idx:04d}"
        new_frame_path = os.path.join(dst_seq, new_frame_name)

        os.makedirs(new_frame_path, exist_ok=True)

        # 폴더 내 파일 복사 및 이름 변경
        old_prefix = old_frame_dir  # e.g., "frame_0005"
        for fname in os.listdir(old_frame_path):
            # 파일 이름에서 old prefix를 new prefix로 교체
            new_fname = fname.replace(old_prefix, new_frame_name)
            src_file = os.path.join(old_frame_path, fname)
            dst_file = os.path.join(new_frame_path, new_fname)
            shutil.copy2(src_file, dst_file)

    new_count = len(selected)
    print(f"  {seq_name}: {total_frames} frames (10hz) -> {new_count} frames (2hz)")


def main():
    # 시퀀스 목록 수집
    sequences = sorted([
        d for d in os.listdir(SRC_DIR)
        if os.path.isdir(os.path.join(SRC_DIR, d)) and d.startswith("SocialACT_")
    ])

    print(f"소스 디렉토리: {SRC_DIR}")
    print(f"출력 디렉토리: {DST_DIR}")
    print(f"샘플링 간격: 매 {SAMPLE_STEP}프레임 (10hz -> 2hz)")
    print(f"시퀀스 수: {len(sequences)}")
    print()

    for seq_name in sequences:
        downsample_sequence(seq_name)

    print()
    print("완료!")


if __name__ == "__main__":
    main()
