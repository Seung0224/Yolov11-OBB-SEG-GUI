# YOLOv11-OBB-SEG Viewer

Windows 환경에서 **Ultralytics YOLOv11 프레임워크**를 활용하여  
- **OBB (Oriented Bounding Box)**  
- **Segmentation (Polygon Masks)**  

두 가지 모드를 모두 지원하는 Python 기반 GUI 툴입니다.  
TensorRT `.engine` 및 PyTorch `.pt` 모델을 모두 실행할 수 있으며,  
Tkinter UI를 통해 이미지 로드 → 추론 → 시각화 → 결과 저장 과정을 간단히 수행할 수 있습니다.

---

<img width="1002" height="732" alt="image" src="https://github.com/user-attachments/assets/889f3b3b-1d2d-4bf7-88bb-1264db9fbefb" />


## 📦 프로젝트 개요

- **플랫폼:** Visual Studio Code (Python 3.11)  
- **프레임워크:** Ultralytics YOLOv11  
- **목적:** 단일 이미지에서 OBB 및 Segmentation 결과를 직관적으로 확인  
- **모델 포맷:** `.pt` (PyTorch), `.engine` (TensorRT)  
- **UI:** Tkinter 기반 GUI  

---

## ✅ 주요 기능

### 1. 📂 모델 로드
- PyTorch `.pt` 모델 또는 TensorRT `.engine` 모델 선택 가능  
- 파일명 자동 분석 → **OBB / SEG 모드** 자동 결정  
  - `*_seg.engine`, `seg.pt` → Segmentation  
  - `*_obb.engine`, `obb.pt` → OBB  

### 2. 🖼️ 이미지 추론
- SEG 모드: **Polygon 마스크 Overlay** + 클래스명 + 점수  
- OBB 모드: **회전 박스(Oriented BBox)** + 클래스명 + 점수  
- 결과물은 항상 원본 이미지 좌표계(`res.orig_img`)에 정렬  

### 3. 🔍 파라미터 조정
- **Confidence (Conf)**  
- **IoU Threshold (IoU)**  
- **Mask Threshold (MaskThr, Seg 전용)**  

### 4. 💾 결과 저장
- 결과 이미지를 PNG/JPG로 저장 가능  
- 실행 중 추론 이미지를 Tkinter 캔버스에 즉시 표시  

### 5. ⚡ 성능 최적화
- TensorRT 엔진 실행 시 CUDA 자동 선택  
- PyTorch `.pt` 실행 시 CPU/GPU 자동 전환  
- Half-precision(FP16) 지원  

---

## 🧰 사용 방법

1. YOLOv11 모델을 학습 후 `.pt` 또는 `.engine` 포맷으로 준비  
   - Segmentation:  
     ```python
     YOLO("your_seg.pt", task="segment").export(format="engine", imgsz=320, half=True, device=0)
     ```
   - OBB:  
     ```python
     YOLO("your_obb.pt", task="obb").export(format="engine", imgsz=320, half=True, device=0)
     ```

2. GUI 실행:
   ```bash
   python ObjectViewer.py

3. 실행 후:
   Load Image 버튼으로 추론할 이미지를 선택, Run Inference 버튼으로 모델 실행, Seg/OBB 결과 Overlay 확인, 필요 시 Save Result 버튼으로 결과 저장

## 🚀 참고사항
 - Accuracy를 높이고싶다면 입력이미지를 640으로 바꾸는것으로 추천드립니다.
 - engine 파일을 만들시 꼭 실행하려는 PC에서 만들어야합니다. GoogleColab이나 다른 환경에서 만들경우 GPU 환경이나 버전 등이 맞지않을수 있기때문에 작동하지 않을수도 있습니다.


| 구성 요소  | 내용                             |
| ------ | ------------------------------ |
| 언어     | Python 3.11                    |
| 프레임워크  | Ultralytics YOLOv11            |
| UI     | Tkinter                        |
| 이미지 처리 | OpenCV-Python                  |
| 시각화    | Pillow                         |
| 가속     | CUDA 12.1, TensorRT 10.x (선택)  |
| 실행 환경  | Visual Studio Code, Windows 10 |

