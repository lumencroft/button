#!/usr/bin/env python3
"""
YOLO11 학습 스크립트 - 깔끔한 버전
- Pretrained 모델 사용
- 공식 Data Augmentation만 사용
- best.pt만 저장 (용량 절약)
"""

import os
from ultralytics import YOLO
import torch

def main():
    """메인 학습 함수"""
    print("🚀 YOLO11 학습 시작 - 깔끔한 버전")
    print("=" * 50)
    
    # GPU 설정
    if torch.cuda.is_available():
        print(f"✅ GPU 사용 가능: {torch.cuda.get_device_name(0)}")
        device = 0  # GPU 사용
    else:
        print("⚠️  GPU 사용 불가, CPU로 학습")
        device = 'cpu'
    
    # 데이터셋 YAML 파일 생성
    create_dataset_yaml()
    
    print("📥 YOLO11 pretrained 모델 로드...")
    model = YOLO('yolo11n.pt')
    
    # 학습 설정
    print("🎯 학습 시작...")
    results = model.train(
        data='data/dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        device=device,
        project='runs/train',
        name='clean_training',
        
        # 공식 Data Augmentation (YOLO/PyTorch 지원)
        hsv_h=0.015,      # HSV-Hue augmentation
        hsv_s=0.7,        # HSV-Saturation augmentation  
        hsv_v=0.4,        # HSV-Value augmentation
        degrees=10.0,     # 회전 (10도)
        translate=0.1,    # 이동 (10%)
        scale=0.5,        # 크기 변화 (50%)
        shear=0.0,        # 기울기
        perspective=0.0,  # 원근 변환
        flipud=0.0,       # 상하 반전
        fliplr=0.5,       # 좌우 반전 (50%)
        mosaic=1.0,       # Mosaic augmentation
        mixup=0.15,       # Mixup augmentation
        copy_paste=0.3,   # Copy-paste augmentation
        
        # 학습률 설정
        lr0=0.01,         # 초기 학습률
        lrf=0.01,         # 최종 학습률
        momentum=0.937,   # 모멘텀
        weight_decay=0.0005,  # 가중치 감쇠
        
        # 저장 설정 (best.pt만 저장)
        save=True,        # 모델 저장 활성화
        save_period=-1,   # 주기적 저장 비활성화 (best.pt만 저장)
        save_txt=False,   # 텍스트 결과 저장 비활성화
        save_conf=False,  # 신뢰도 저장 비활성화
        save_crop=False,  # 크롭 이미지 저장 비활성화
        
        # 기타 설정
        patience=50,      # Early stopping patience
        val=True,         # 검증 실행
        plots=True,       # 그래프 생성
        verbose=True,     # 상세 출력
    )
    
    print("🎉 학습 완료!")
    print(f"📁 결과 저장 위치: {results.save_dir}")
    print(f"🏆 최고 모델: {results.save_dir}/weights/best.pt")

def create_dataset_yaml():
    """데이터셋 YAML 파일 생성"""
    yaml_content = """# YOLO11 Dataset Configuration - Clean
path: data  # dataset root dir
train: images  # train images (relative to 'path')
val: images    # val images (relative to 'path')

# Classes
nc: 12  # number of classes
names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'up', 'down']  # class names
"""
    
    # YAML 파일 저장 (상대 경로)
    yaml_path = 'data/dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"📄 데이터셋 YAML 생성: {yaml_path}")

if __name__ == "__main__":
    main()