#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Dataset Maker (OBB + Segmentation 통합 버전)
- 이미지/라벨 쌍을 모아 train/val 분할 후 data.yaml 생성
- 실행 시 모드 선택 가능: obb / seg

사용법:
    python make_yolo_dataset.py obb
    python make_yolo_dataset.py seg
"""

import os, sys, random, shutil, yaml
from pathlib import Path

# ===== 사용자 경로/설정 =====
InitPath = r"D:\TOYPROJECT\OS\Model\SEG" 
SRC = InitPath + "\Init"      # 원본 폴더
DST = InitPath + "\RESULTMODEL"  # 결과 폴더
VAL_RATIO = 0.1
SEED = 0
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 클래스 이름:
#   1) 아래 NAMES를 직접 적거나
#   2) None으로 두면 SRC/classes.txt 자동 탐색
NAMES = None   # 예: ["chip"]  또는  None(자동)

random.seed(SEED)


def load_names(src_dir: Path):
    """classes.txt 읽거나 직접 정의된 NAMES 사용"""
    if isinstance(NAMES, list) and len(NAMES) > 0:
        return NAMES
    cand = src_dir / "classes.txt"
    if cand.exists():
        with open(cand, "r", encoding="utf-8") as f:
            names = [ln.strip() for ln in f if ln.strip()]
        if names:
            return names
    return ["chip"]


def looks_like_segment_label(txt_path: Path) -> bool:
    """세그멘테이션 포맷 점검"""
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                # YOLO 세그: class + 짝수 좌표 → 총 토큰 수는 홀수, 최소 7개 이상
                if len(parts) < 7 or (len(parts) % 2 == 0):
                    return False
        return True
    except Exception:
        return False


def make_dataset(task: str):
    """OBB 또는 Segmentation 데이터셋 생성"""
    src_path = Path(SRC)
    dst_path = Path(DST)

    image_by_stem = {}
    label_by_stem = {}
    for p in src_path.rglob("*"):
        if p.is_file():
            if p.suffix.lower() in IMG_EXTS:
                image_by_stem[p.stem] = p
            elif p.suffix.lower() == ".txt" and p.name.lower() != "classes.txt":
                label_by_stem[p.stem] = p

    pairs = [(ipath, label_by_stem[stem]) for stem, ipath in image_by_stem.items()
             if stem in label_by_stem and label_by_stem[stem].is_file()]

    if not pairs:
        raise SystemExit("이미지-라벨 쌍을 찾지 못했습니다. SRC 경로와 파일 구성을 확인하세요.")

    # 세그멘테이션 모드일 때만 포맷 점검
    if task == "segment":
        bad = [p for _, p in pairs if not looks_like_segment_label(p)]
        if bad:
            print(f"[경고] 세그멘테이션 포맷이 아닌 라벨 파일 {len(bad)}개가 있습니다. (예: {bad[0].name})")

    # 결과 폴더 구조 생성
    for d in ["images/train", "images/val", "labels/train", "labels/val"]:
        (dst_path / d).mkdir(parents=True, exist_ok=True)

    # 셔플 및 분할
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * VAL_RATIO))
    val_set = set(pairs[:n_val])

    # 복사
    for img, lab in pairs:
        subset = "val" if (img, lab) in val_set else "train"
        shutil.copy2(img, dst_path / "images" / subset / img.name)
        shutil.copy2(lab, dst_path / "labels" / subset / lab.name)

    # data.yaml 생성
    names = load_names(src_path)
    data = {
        "path": str(dst_path.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(names)},
        "task": task
    }
    with open(dst_path / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    print("완료!")
    print(f"총 샘플: {len(pairs)}  → train: {len(pairs)-n_val}, val: {n_val}")
    print(f"결과 폴더: {dst_path}")
    print(f"data.yaml: {dst_path/'data.yaml'}")
    print(f"클래스: {names}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in {"obb", "seg"}:
        print("사용법: python make_yolo_dataset.py [obb|seg]")
        sys.exit(1)
    make_dataset(sys.argv[1])
