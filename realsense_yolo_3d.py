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
import threading
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
        
        # UDP 수신 설정 (ready 신호 받기)
        self.udp_receiver = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.receiver_ip = "0.0.0.0"  # 모든 인터페이스에서 수신
        self.receiver_port = 5006
        self.udp_receiver.bind((self.receiver_ip, self.receiver_port))
        
        # BUTTON_POSITION 프로토콜 설정 (psock 구조 참고)
        self.start_byte = 'POLA'
        self.message_id = 102  # BUTTON_POSITION
        self.length = 16  # 명세에 따라 16으로 변경
        
        # 상태 관리
        self.state = "waiting_ready"  # waiting_ready, looking_up, looking_7, completed
        self.ready_signal_received = False
        self.up_button_sent = False
        self.button_7_sent = False
        
        # 버튼 감지 시 전송용 타이머
        self.last_send_time = 0
        self.send_interval = 1.0  # 1초마다 전송
        
        # 스레드 제어
        self.running = True
        self.udp_thread = None
        

        
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
    
    def udp_receiver_thread(self):
        """UDP 수신 스레드 - ready 신호를 받아서 상태 변경"""
        print(f"📡 UDP 수신 시작: {self.receiver_ip}:{self.receiver_port}")
        
        while self.running:
            try:
                # BPM_INFO 프로토콜 수신 (16바이트)
                data, addr = self.udp_receiver.recvfrom(16)
                
                if len(data) == 16:
                    # 프로토콜 파싱
                    unpacked = struct.unpack("<4s2H8B", data)
                    start_byte = unpacked[0].decode()
                    message_id = unpacked[1]
                    length = unpacked[2]
                    activation = unpacked[3]
                    button_press_done = unpacked[4]
                    button_status = unpacked[5]
                    ready = unpacked[6]
                    operate_tray_door = unpacked[7]
                    
                    # BPM_INFO 메시지이고 ready 신호가 1인 경우
                    if start_byte == 'POLA' and message_id == 103 and ready == 1:
                        print(f"🎯 Ready 신호 수신! (from {addr[0]}:{addr[1]})")
                        self.ready_signal_received = True
                        self.state = "looking_up"
                        self.up_button_sent = False
                        self.button_7_sent = False
                        
            except Exception as e:
                if self.running:  # 정상 종료가 아닌 경우에만 에러 출력
                    print(f"❌ UDP 수신 오류: {e}")
                break
    
    def start_udp_receiver(self):
        """UDP 수신 스레드 시작"""
        self.udp_thread = threading.Thread(target=self.udp_receiver_thread, daemon=True)
        self.udp_thread.start()
    
    def send_button_position(self, x_3d, y_3d, z_3d, button_type="up"):
        """버튼 위치를 UDP로 전송"""
        try:
            # 현재 시간
            current_time = time.time()
            
            # BUTTON_POSITION 프로토콜에 맞는 패킷 생성 (length=16으로 변경)
            # 헤더: start_byte(4) + message_id(2) + length(2) = 8 bytes
            # 페이로드: time(8) = 8 bytes
            # 총 16 bytes
            
            message = struct.pack(
                "<4s2Hd",  # 포맷: 4s(start_byte) + 2H(message_id, length) + d(time)
                self.start_byte.encode(),  # start_byte: 'POLA'
                self.message_id,           # message_id: 102
                self.length,               # length: 16
                current_time               # time: double
            )
            
            # UDP 전송
            self.udp_socket.sendto(message, (self.target_ip, self.target_port))
            print(f"📡 UDP 전송 완료: {button_type.upper()} 버튼 위치 ({x_3d:.1f}, {y_3d:.1f}, {z_3d:.1f})mm -> {self.target_ip}:{self.target_port}")
            
        except Exception as e:
            print(f"❌ UDP 전송 실패: {e}")
    
    def run(self):
        """메인 실행 루프"""
        if not self.start_camera():
            return
        
        # UDP 수신 스레드 시작
        self.start_udp_receiver()
        
        # 카메라 매트릭스 가져오기
        K_color, K_depth, dist_coeffs_color, dist_coeffs_depth = self.coord_calculator.get_camera_matrices(
            self.color_intrinsics, self.depth_intrinsics
        )
        
        print("\n🎥 RealSense + YOLO + 3D 좌표 계산 시작!")
        print("🎯 Ready 신호 대기 중... (UDP 포트 5006에서 수신)")
        print("🎯 Ready 신호 수신 시: UP 버튼 → 7 버튼 순서로 감지")
        print(f"📡 전송 대상: {self.target_ip}:{self.target_port}")
        print("\n⌨️  키보드 입력:")
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
                
                # 상태에 따른 타겟 버튼 결정
                target_class_id = None
                target_class_name = None
                
                if self.state == "looking_up":
                    target_class_id = 10  # 'up' 버튼
                    target_class_name = "up"
                elif self.state == "looking_7":
                    target_class_id = 7   # '7' 버튼
                    target_class_name = "7"
                
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
                            
                            # 3D 좌표 계산
                            x_3d, y_3d, z_3d = self.coord_calculator.get_3d_coordinates(
                                (x1, y1, x2, y2), depth_image, K_depth, dist_coeffs_depth, K_color, dist_coeffs_color
                            )
                            
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
                                text_3d = f"3D: ({x_3d:.1f}, {y_3d:.1f}, {z_3d:.1f})mm"
                                
                                cv2.putText(annotated_image, text, (x1, y1-30), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                cv2.putText(annotated_image, text_3d, (x1, y1-15), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                                
                                # 중심점에 원 그리기
                                cv2.circle(annotated_image, (center_x, center_y), 3, (0, 0, 255), -1)
                                
                                # 타겟 버튼 감지 시 UDP 전송
                                if (target_class_id is not None and class_id == target_class_id and 
                                    not (self.state == "looking_up" and self.up_button_sent) and
                                    not (self.state == "looking_7" and self.button_7_sent)):
                                    
                                    current_time = time.time()
                                    if current_time - self.last_send_time >= self.send_interval:
                                        self.send_button_position(x_3d, y_3d, z_3d, target_class_name)
                                        self.last_send_time = current_time
                                        
                                        if self.state == "looking_up":
                                            self.up_button_sent = True
                                            self.state = "looking_7"
                                            print("🎯 UP 버튼 전송 완료! 이제 7 버튼을 찾는 중...")
                                        elif self.state == "looking_7":
                                            self.button_7_sent = True
                                            self.state = "completed"
                                            print("🎯 7 버튼 전송 완료! 모든 작업 완료!")
                            else:
                                text = f"{class_name}: {confidence:.2f} (No depth)"
                                cv2.putText(annotated_image, text, (x1, y1-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Depth 이미지 시각화 (컬러맵)
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
                )
                
                # 현재 상태 정보를 화면에 표시
                if self.state == "waiting_ready":
                    status_info = "상태: Ready 신호 대기 중..."
                    color = (0, 255, 255)  # 노란색
                elif self.state == "looking_up":
                    status_info = "상태: UP 버튼 찾는 중..."
                    color = (0, 255, 0)    # 초록색
                elif self.state == "looking_7":
                    status_info = "상태: 7 버튼 찾는 중..."
                    color = (255, 0, 0)    # 파란색
                elif self.state == "completed":
                    status_info = "상태: 모든 작업 완료!"
                    color = (0, 255, 0)    # 초록색
                else:
                    status_info = f"상태: {self.state}"
                    color = (255, 255, 255)  # 흰색
                
                cv2.putText(annotated_image, status_info, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)  # 검은색 테두리
                cv2.putText(annotated_image, status_info, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
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
                elif key == ord('c'):  # 'c' 키: 왜곡 보정 토글
                    self.coord_calculator.use_undistortion = not self.coord_calculator.use_undistortion
                    status = "ON" if self.coord_calculator.use_undistortion else "OFF"
                    print(f"\n🔧 왜곡 보정 토글: {status}")
                elif key == ord('m'):  # 'm' 키: 3D 계산 매트릭스 토글
                    self.coord_calculator.use_color_matrix_for_3d = not self.coord_calculator.use_color_matrix_for_3d
                    status = "Color" if self.coord_calculator.use_color_matrix_for_3d else "Depth"
                    print(f"\n🎯 3D 계산 매트릭스 토글: {status}")
                elif key == ord('r'):  # 'r' 키: 상태 리셋
                    self.state = "waiting_ready"
                    self.up_button_sent = False
                    self.button_7_sent = False
                    print(f"\n🔄 상태 리셋: Ready 신호 대기 중...")
                    
        except KeyboardInterrupt:
            print("\n사용자에 의해 중단되었습니다.")
        
        finally:
            # 정리
            self.running = False
            self.pipeline.stop()
            self.udp_socket.close()
            self.udp_receiver.close()
            cv2.destroyAllWindows()
            print("✅ 프로그램이 종료되었습니다.")

def main():
    """메인 함수"""
    print("RealSense + YOLO + 3D 좌표 계산 프로그램")
    print("=" * 50)
    
    # 모델 경로 확인
    model_path = "runs/train/clean_training2/weights/best.pt"
    
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
