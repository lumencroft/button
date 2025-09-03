# Button Detection with Depth Camera

버튼 인식 및 depth 카메라를 이용한 3D 좌표 변환 프로젝트

## 기능

1. **이미지에서 버튼 인식**: `button_detection_200epochs` 모델 사용
2. **Depth 카메라 결과로 3D 좌표 변환**: x, y, z 좌표를 m 단위로 계산

## 프로젝트 구조

```
pevrecog/
├── main.py                          # 메인 클래스 (통합 기능)
├── button_detection.py              # 버튼 검출 클래스
├── coordinate_calculation.py        # 3D 좌표 계산 클래스
├── example_usage.py                 # 사용 예제
├── realsense_capture.py             # RealSense 카메라 캡처
├── realsense_yolo_inference.py      # RealSense + YOLO 추론
├── config/
│   └── floor_config.yaml           # 설정 파일
├── data/                            # 데이터셋
│   ├── auto_dataset/               # 자동 생성 데이터셋
│   ├── final_dataset_collection/   # 최종 수집 데이터셋
│   ├── datasets/                   # 처리된 YOLO 데이터셋
│   └── yolo_dataset*/              # YOLO 훈련용 데이터셋
├── models/                          # 모델 관련
│   ├── trained_models/             # 훈련된 모델들
│   ├── runs_final/                 # 최종 훈련 결과
│   └── legacy/                     # 기존 모델 코드
├── scripts/                         # 데이터 처리 및 훈련 스크립트
│   ├── batch_*.py                  # 배치 처리 스크립트
│   ├── train_*.py                  # 훈련 스크립트
│   ├── interactive_*.py            # 인터랙티브 수정 도구
│   └── prepare_*.py                # 데이터셋 준비 도구
├── tests/                          # 테스트 파일들
├── recognition/                    # 기존 버튼 인식 코드
│   ├── __init__.py
│   └── button_recog.py
└── requirements_realsense.txt      # RealSense 의존성
```

## 설치 및 사용

### 1. 의존성 설치

```bash
pip install torch torchvision ultralytics opencv-python numpy pyyaml
```

### 2. 기본 사용법

```python
from main import ButtonDetectionWithDepth

# 버튼 검출기 초기화
detector = ButtonDetectionWithDepth()

# 카메라 파라미터 설정 (실제 카메라 값으로 수정 필요)
detector.set_camera_params(
    fx=500.0,  # 초점거리 x
    fy=500.0,  # 초점거리 y
    cx=320.0,  # 주점 x
    cy=240.0,  # 주점 y
    depth_scale=0.001  # mm to m
)

# 이미지 처리
result = detector.process_image(color_image, depth_image, target_button_class=0)

# 결과 확인
if result['success']:
    print(f"3D 좌표: {result['3d_coords']}")  # [x, y, z] in meters
    print(f"바운딩 박스: {result['bbox']}")    # [x, y, w, h]
```

### 3. 개별 클래스 사용법

```python
from button_detection import ButtonDetection
from coordinate_calculation import CoordinateCalculation

# 버튼 검출
button_detector = ButtonDetection()
bbox = button_detector.predict_coordinates(0, color_image)

# 3D 좌표 계산
coord_calculator = CoordinateCalculation()
coords_3d = coord_calculator.get_button_center_3d(bbox, depth_image)
```

## 카메라 파라미터 설정

실제 카메라의 내부 파라미터를 설정해야 정확한 3D 좌표를 얻을 수 있습니다:

```python
# RealSense 카메라 예제
detector.set_camera_params(
    fx=615.0,   # 실제 초점거리 x
    fy=615.0,   # 실제 초점거리 y
    cx=320.0,   # 실제 주점 x
    cy=240.0,   # 실제 주점 y
    depth_scale=0.001  # mm to m 변환
)
```

## 모델 정보

- **모델**: `button_detection_200epochs`
- **경로**: `models/trained_models/button_detection_200epochs/weights/best.pt`
- **기반**: YOLOv5
- **용도**: 엘리베이터 버튼 검출

## 예제 실행

```bash
python example_usage.py
```

## 주의사항

1. **카메라 파라미터**: 실제 카메라의 내부 파라미터를 정확히 설정해야 합니다
2. **Depth 스케일**: depth 이미지의 단위에 맞게 `depth_scale`을 조정하세요
3. **모델 경로**: 모델 파일이 올바른 경로에 있는지 확인하세요
4. **UDP 통신**: 이 프로젝트는 UDP 통신 기능을 제거했습니다. 다른 프로젝트에서 import하여 사용하세요

## 라이선스

이 프로젝트는 기존 pevrecog 프로젝트에서 필요한 기능만 추출한 버전입니다.
