#!/usr/bin/env python3
"""
RealSense 카메라 + YOLO + 3D 좌표 계산
RealSense 카메라에서 RGB/Depth 스트림을 받아서 YOLO로 inference하고
왜곡 보정을 고려하여 bounding box의 3D 좌표를 계산합니다.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import json
from datetime import datetime
import socket
import struct
import time
from coordinate_calculator import CoordinateCalculator

class RealSenseYOLO3D:
    def __init__(self, model_path=""):
        """RealSense + YOLO + 3D 좌표 계산 클래스 초기화"""
        
        # YOLO 모델 로드
        print(f"YOLO 모델 로딩: {model_path}")
        self.model = YOLO(model_path)
        
        # RealSense 파이프라인 설정
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # 스트림 설정
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # 카메라 내부 파라미터 (이전에 추출한 값들)
        self.color_intrinsics = None
        self.depth_intrinsics = None
        self.depth_scale = None
        
        # 클래스 이름
        self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'up', 'down']
        
        # 좌표 계산기 초기화
        self.coord_calculator = None
        
        # UDP 전송 설정
        self.udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.target_ip = "192.168.1.131"
        self.target_port = 5005
        
        # BUTTON_POSITION 프로토콜 설정 (psock 구조 참고)
        self.start_byte = 'POLA'
        self.message_id = 102  # BUTTON_POSITION
        self.length = 32  # 헤더 8 + 페이로드 24
        
        # 버튼 감지 시 전송용 타이머
        self.last_send_time = 0
        self.send_interval = 2.0  # 2초마다 전송
        self.target_button_detected = False
        self.target_class_id = 10  # 기본값: 'up' 버튼 (클래스 ID 10)
        self.target_class_name = 'up'
        

        
    def start_camera(self):
        """RealSense 카메라 시작"""
        try:
            # 파이프라인 시작
            profile = self.pipeline.start(self.config)
            
            # 내부 파라미터 가져오기
            color_stream = profile.get_stream(rs.stream.color)
            self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
            depth_stream = profile.get_stream(rs.stream.depth)
            self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
            
            print("✅ RealSense 카메라 연결 성공!")
            print(f"Color 해상도: {self.color_intrinsics.width}x{self.color_intrinsics.height}")
            print(f"Depth 해상도: {self.depth_intrinsics.width}x{self.depth_intrinsics.height}")
            
            # Depth 단위 확인
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            print(f"Depth 단위 스케일: {self.depth_scale}")
            print(f"실제 depth 값 = raw_depth * {self.depth_scale} (미터 단위)")
            
            # 좌표 계산기 초기화
            self.coord_calculator = CoordinateCalculator(self.depth_scale)
            
            return True
            
        except Exception as e:
            print(f"❌ RealSense 카메라 연결 실패: {e}")
            return False
    

    
    def send_button_position(self, x_3d, y_3d, z_3d):
        """버튼 위치를 UDP로 전송"""
        try:
            # 현재 시간
            current_time = time.time()
            
            # BUTTON_POSITION 프로토콜에 맞는 패킷 생성
            # 헤더: start_byte(4) + message_id(2) + length(2) = 8 bytes
            # 페이로드: time(8) + button_pos[3](12) + type(1) + reserved[3](3) = 24 bytes
            # 총 32 bytes
            
            message = struct.pack(
                "<4s2Hd3fB3B",  # 포맷: 4s(start_byte) + 2H(message_id, length) + d(time) + 3f(button_pos) + B(type) + 3B(reserved)
                self.start_byte.encode(),  # start_byte: 'POLA'
                self.message_id,           # message_id: 102
                self.length,               # length: 32
                current_time,              # time: double
                float(x_3d),               # button_pos[0]: float
                float(y_3d),               # button_pos[1]: float  
                float(z_3d),               # button_pos[2]: float
                1,                         # type: 1 (camera coordinate)
                0, 0, 0                    # reserved[3]: 0으로 채움
            )
            
            # UDP 전송
            self.udp_socket.sendto(message, (self.target_ip, self.target_port))
            print(f"📡 UDP 전송 완료: UP 버튼 위치 ({x_3d:.1f}, {y_3d:.1f}, {z_3d:.1f})mm -> {self.target_ip}:{self.target_port}")
            
        except Exception as e:
            print(f"❌ UDP 전송 실패: {e}")
    
    def run(self):
        """메인 실행 루프"""
        if not self.start_camera():
            return
        
        # 카메라 매트릭스 가져오기
        K_color, K_depth, dist_coeffs_color, dist_coeffs_depth = self.coord_calculator.get_camera_matrices(
            self.color_intrinsics, self.depth_intrinsics
        )
        
        print("\n🎥 RealSense + YOLO + 3D 좌표 계산 시작!")
        print("🎯 현재 타겟 버튼: {self.target_class_name} (클래스 ID: {self.target_class_id})")
        print("🎯 감지 시 2초마다 실제 좌표를 UDP로 전송합니다.")
        print(f"📡 전송 대상: {self.target_ip}:{self.target_port}")
        print("\n⌨️  키보드 입력:")
        print("   숫자 키 (0-9): 해당 숫자 버튼으로 변경")
        print("   'u' 키: 'up' 버튼으로 변경")
        print("   'd' 키: 'down' 버튼으로 변경")
        print("   'c' 키: 왜곡 보정 토글")
        print("   'm' 키: 3D 계산 매트릭스 토글 (Color/Depth)")
        print("   ESC 키: 종료")
        print("-" * 50)
        
        try:
            while True:
                # 프레임 가져오기
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                

                
                # 이미지 변환
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Color 이미지 왜곡 보정
                color_undistorted = self.coord_calculator.undistort_image(color_image, K_color, dist_coeffs_color)
                
                # YOLO inference
                results = self.model(color_undistorted, verbose=False)
                
                # 결과 시각화
                annotated_image = color_undistorted.copy()
                
                # 타겟 버튼 감지 상태 초기화
                target_detected_this_frame = False
                
                # 각 detection에 대해 3D 좌표 계산
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Bounding box 좌표
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            confidence = box.conf[0].cpu().numpy()
                            if confidence < 0.5:
                                continue
                            class_id = int(box.cls[0].cpu().numpy())
                            
                            # 클래스 이름 가져오기
                            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                            
                            # 픽셀 좌표 계산
                            print(f"\n🔍 {class_name} 버튼 감지 - Confidence: {confidence:.2f}")
                            x_3d, y_3d, z_3d = self.coord_calculator.get_3d_coordinates(
                                (x1, y1, x2, y2), depth_image, K_depth, dist_coeffs_depth, K_color, dist_coeffs_color
                            )
                            
                            # 시각화
                            
                            # Bounding box 그리기
                            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # 텍스트 정보
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            
                            if x_3d is not None:
                                # Depth 값 확인을 위한 디버깅 정보
                                center_depth_raw = depth_image[center_y, center_x]
                                center_depth_meters = center_depth_raw * self.depth_scale if self.depth_scale else center_depth_raw
                                
                                text = f"{class_name}: {confidence:.2f}"
                                
                                # 완전한 3D 좌표 표시
                                text_3d = f"3D: ({x_3d:.1f}, {y_3d:.1f}, {z_3d:.1f})mm"
                                text_debug = f"Raw: {center_depth_raw}, Meters: {center_depth_meters:.3f}m"
                                
                                cv2.putText(annotated_image, text, (x1, y1-30), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                cv2.putText(annotated_image, text_3d, (x1, y1-15), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                                cv2.putText(annotated_image, text_debug, (x1, y1), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                                
                                # 중심점에 원 그리기
                                cv2.circle(annotated_image, (center_x, center_y), 3, (0, 0, 255), -1)
                                
                                # 타겟 버튼 감지 시 UDP 전송
                                if class_id == self.target_class_id:  # confidence 임계값 설정
                                    target_detected_this_frame = True
                                    current_time = time.time()
                                    if current_time - self.last_send_time >= self.send_interval:
                                        self.send_button_position(x_3d, y_3d, z_3d)
                                        self.last_send_time = current_time
                                        self.target_button_detected = True
                            else:
                                text = f"{class_name}: {confidence:.2f} (No depth)"
                                cv2.putText(annotated_image, text, (x1, y1-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # 타겟 버튼이 감지되지 않았으면 상태 리셋
                if not target_detected_this_frame:
                    self.target_button_detected = False
                
                # Depth 이미지 시각화 (컬러맵)
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
                )
                
                # 현재 타겟 버튼 정보를 화면에 표시
                target_info = f"Target: {self.target_class_name} (ID: {self.target_class_id})"
                cv2.putText(annotated_image, target_info, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_image, target_info, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
                
                # 왜곡 보정 상태 표시
                distortion_info = f"Distortion: {'ON' if self.coord_calculator.use_undistortion else 'OFF'}"
                cv2.putText(annotated_image, distortion_info, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_image, distortion_info, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.coord_calculator.use_undistortion else (0, 0, 255), 1)
                
                # 3D 계산 매트릭스 상태 표시
                matrix_info = f"3D Matrix: {'Color' if self.coord_calculator.use_color_matrix_for_3d else 'Depth'}"
                cv2.putText(annotated_image, matrix_info, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_image, matrix_info, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0) if self.coord_calculator.use_color_matrix_for_3d else (255, 165, 0), 1)
                
                # 화면에 표시
                cv2.imshow('RealSense + YOLO + 3D', annotated_image)
                cv2.imshow('Depth', depth_colormap)
                
                # 키보드 입력 처리
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    break
                elif key == ord('u'):  # 'u' 키: up 버튼
                    self.target_class_id = 10
                    self.target_class_name = 'up'
                    print(f"\n🎯 타겟 버튼 변경: {self.target_class_name} (클래스 ID: {self.target_class_id})")
                elif key == ord('d'):  # 'd' 키: down 버튼
                    self.target_class_id = 11
                    self.target_class_name = 'down'
                    print(f"\n🎯 타겟 버튼 변경: {self.target_class_name} (클래스 ID: {self.target_class_id})")
                elif key == ord('c'):  # 'c' 키: 왜곡 보정 토글
                    self.coord_calculator.use_undistortion = not self.coord_calculator.use_undistortion
                    status = "ON" if self.coord_calculator.use_undistortion else "OFF"
                    print(f"\n🔧 왜곡 보정 토글: {status}")
                elif key == ord('m'):  # 'm' 키: 3D 계산 매트릭스 토글
                    self.coord_calculator.use_color_matrix_for_3d = not self.coord_calculator.use_color_matrix_for_3d
                    status = "Color" if self.coord_calculator.use_color_matrix_for_3d else "Depth"
                    print(f"\n🎯 3D 계산 매트릭스 토글: {status}")
                elif ord('0') <= key <= ord('9'):  # 숫자 키 (0-9)
                    self.target_class_id = key - ord('0')
                    self.target_class_name = str(self.target_class_id)
                    print(f"\n🎯 타겟 버튼 변경: {self.target_class_name} (클래스 ID: {self.target_class_id})")
                    
        except KeyboardInterrupt:
            print("\n사용자에 의해 중단되었습니다.")
        
        finally:
            # 정리
            self.pipeline.stop()
            self.udp_socket.close()
            cv2.destroyAllWindows()
            print("✅ 프로그램이 종료되었습니다.")

def main():
    """메인 함수"""
    print("RealSense + YOLO + 3D 좌표 계산 프로그램")
    print("=" * 50)
    
    # 모델 경로 확인
    model_path = "runs/train/clean_training/weights/best.pt"
    
    try:
        app = RealSenseYOLO3D(model_path)
        app.run()
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("다음을 확인해주세요:")
        print("1. RealSense 카메라가 연결되어 있는지")
        print("2. YOLO 모델 파일이 존재하는지")
        print("3. 필요한 라이브러리가 설치되어 있는지")

if __name__ == "__main__":
    main()
