#!/usr/bin/env python3
"""
엘리베이터 버튼 감지 및 3D 좌표 전송 모듈
HMI에서 목적지층을 받고, RealSense 카메라로 버튼을 감지하여 3D 좌표를 UDP로 전송합니다.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import socket
import struct
import time
import yaml
import logging
import os
import sys
from ultralytics import YOLO
from typing import Optional, Tuple, Dict, Any

# psock 모듈 import
home_path = os.path.expanduser("~")
sys.path.append(f"{home_path}/ws")
from psock.udp.udp_tx import UdpTx
from psock.udp.udp_rx_delivery_service import UdpRxDeliveryService

class ElevatorButtonDetector:
    def __init__(self, model_path="yolo11n.pt"):
        """엘리베이터 버튼 감지기 초기화"""
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # YOLO 모델 로드
        self.logger.info(f"YOLO 모델 로딩: {model_path}")
        self.model = YOLO(model_path)
        
        # 클래스 이름 (숫자 0-9, up, down)
        self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'up', 'down']
        
        # RealSense 카메라 설정
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # 카메라 내부 파라미터
        self.color_intrinsics = None
        self.depth_intrinsics = None
        self.depth_scale = None
        
        # UDP 통신 설정 (psock 모듈 사용)
        self.udp_tx = UdpTx()
        self.setup_udp()
        
        # 상태 변수
        self.current_floor = 1  # 현재 층수
        self.target_floor = 1   # 목적지 층수
        self.is_delivery_active = False
        self.is_robot_inside_elevator = False
        
    def setup_udp(self):
        """UDP 통신 설정 (psock 모듈 사용)"""
        # UDP 수신 소켓 생성 (HMI에서 DELIVERY_INFO를 받기 위해)
        self.udp_receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
        self.udp_receiver.bind(("", 5001))  # UDP_DELIVERY_SERVICE 포트
        self.udp_receiver.settimeout(0.1)  # 100ms 타임아웃
        
        # psock에서 IP 주소와 포트 가져오기
        self.bpm_ip = self.udp_tx.IP_ADDRESS_PORT["UDP_BPM"]["ip_address"]
        self.bpm_port = self.udp_tx.IP_ADDRESS_PORT["UDP_BPM"]["port"]
        self.delivery_service_ip = self.udp_tx.IP_ADDRESS_PORT["UDP_DELIVERY_SERVICE"]["ip_address"]
        self.delivery_service_port = self.udp_tx.IP_ADDRESS_PORT["UDP_DELIVERY_SERVICE"]["port"]
        
        self.logger.info("UDP 통신 설정 완료 (psock 모듈 사용)")
        
    def start_camera(self) -> bool:
        """RealSense 카메라 시작"""
        try:
            profile = self.pipeline.start(self.config)
            
            # 내부 파라미터 가져오기
            color_stream = profile.get_stream(rs.stream.color)
            self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
            depth_stream = profile.get_stream(rs.stream.depth)
            self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
            
            # Depth 단위 스케일
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            self.logger.info("✅ RealSense 카메라 연결 성공!")
            self.logger.info(f"Color 해상도: {self.color_intrinsics.width}x{self.color_intrinsics.height}")
            self.logger.info(f"Depth 해상도: {self.depth_intrinsics.width}x{self.depth_intrinsics.height}")
            self.logger.info(f"Depth 단위 스케일: {self.depth_scale}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ RealSense 카메라 연결 실패: {e}")
            return False
    
    def get_camera_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """카메라 매트릭스 생성"""
        if self.color_intrinsics is None or self.depth_intrinsics is None:
            return None, None, None, None
            
        # Color 카메라 매트릭스
        K_color = np.array([
            [self.color_intrinsics.fx, 0, self.color_intrinsics.ppx],
            [0, self.color_intrinsics.fy, self.color_intrinsics.ppy],
            [0, 0, 1]
        ])
        
        # Depth 카메라 매트릭스
        K_depth = np.array([
            [self.depth_intrinsics.fx, 0, self.depth_intrinsics.ppx],
            [0, self.depth_intrinsics.fy, self.depth_intrinsics.ppy],
            [0, 0, 1]
        ])
        
        # 왜곡 계수
        dist_coeffs_color = np.array(self.color_intrinsics.coeffs)
        dist_coeffs_depth = np.array(self.depth_intrinsics.coeffs)
        
        return K_color, K_depth, dist_coeffs_color, dist_coeffs_depth
    
    def undistort_image(self, image: np.ndarray, K: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
        """이미지 왜곡 보정"""
        return cv2.undistort(image, K, dist_coeffs)
    
    def get_3d_coordinates(self, bbox: Tuple[int, int, int, int], depth_image: np.ndarray, 
                          K_depth: np.ndarray, dist_coeffs_depth: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Bounding box의 3D 좌표 계산"""
        x1, y1, x2, y2 = bbox
        
        # Bounding box 중심점
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # 중심점 주변의 depth 값들 샘플링 (노이즈 제거)
        depth_samples = []
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                sample_x = center_x + dx
                sample_y = center_y + dy
                if 0 <= sample_x < depth_image.shape[1] and 0 <= sample_y < depth_image.shape[0]:
                    depth_val = depth_image[sample_y, sample_x]
                    if depth_val > 0:  # 유효한 depth 값만
                        depth_samples.append(depth_val)
        
        if not depth_samples:
            return None
        
        # 중간값 사용 (노이즈 제거)
        depth_raw = np.median(depth_samples)
        
        # RealSense depth 단위 변환 (미터 단위로 변환 후 mm로 변환)
        if self.depth_scale is not None:
            depth_meters = depth_raw * self.depth_scale
            depth_mm = depth_meters * 1000  # 미터를 mm로 변환
        else:
            depth_mm = depth_raw  # 스케일이 없으면 raw 값 사용
        
        # 픽셀 좌표를 3D 좌표로 변환
        # 왜곡 보정된 픽셀 좌표 사용
        undistorted_point = cv2.undistortPoints(
            np.array([[[center_x, center_y]]], dtype=np.float32),
            K_depth, dist_coeffs_depth
        )[0][0]
        
        u_undist = undistorted_point[0]
        v_undist = undistorted_point[1]
        
        # 3D 좌표 계산 (mm 단위)
        x_3d = (u_undist - K_depth[0, 2]) * depth_mm / K_depth[0, 0]
        y_3d = (v_undist - K_depth[1, 2]) * depth_mm / K_depth[1, 1]
        z_3d = depth_mm
        
        return x_3d, y_3d, z_3d
    
    def send_button_position(self, button_pos: Tuple[float, float, float], tooltip_pos: Tuple[float, float, float]):
        """버튼 위치를 UDP로 전송 (psock 모듈 사용)"""
        try:
            # psock 모듈의 tx_button_position 함수 사용
            self.udp_tx.tx_button_position(
                button_pos=button_pos,
                tooltip_pos=tooltip_pos
            )
            
            button_x, button_y, button_z = button_pos
            tooltip_x, tooltip_y, tooltip_z = tooltip_pos
            
            self.logger.info(f"버튼 위치 전송: Button({button_x:.1f}, {button_y:.1f}, {button_z:.1f})mm, "
                           f"Tooltip({tooltip_x:.1f}, {tooltip_y:.1f}, {tooltip_z:.1f})mm")
            
        except Exception as e:
            self.logger.error(f"버튼 위치 전송 실패: {e}")
    
    def send_delivery_info(self, start_to_deliver: int = 0, current_floor: int = None, 
                          target_floor: int = None, robot_location: int = 0):
        """배송 정보를 UDP로 전송 (psock 모듈 사용)"""
        try:
            if current_floor is None:
                current_floor = self.current_floor
            if target_floor is None:
                target_floor = self.target_floor
                
            # psock 모듈의 tx_delivery_info 함수 사용
            self.udp_tx.tx_delivery_info(
                start_to_deliver=start_to_deliver,
                current_floor=current_floor,
                target_floor=target_floor,
                robot_location=robot_location
            )
            
            self.logger.info(f"배송 정보 전송: Start={start_to_deliver}, Current={current_floor}, "
                           f"Target={target_floor}, Location={robot_location}")
            
        except Exception as e:
            self.logger.error(f"배송 정보 전송 실패: {e}")
    
    def send_bpm_info(self, activation: int = 0, button_press_done: int = 0, 
                     button_status: int = 0, ready: int = 0, operate_tray_door: int = 0):
        """BPM 정보를 UDP로 전송 (psock 모듈 사용)"""
        try:
            # psock 모듈의 tx_bpm_info 함수 사용
            self.udp_tx.tx_bpm_info(
                activation=activation,
                button_press_done=button_press_done,
                button_status=button_status,
                ready=ready,
                operate_tray_door=operate_tray_door
            )
            
            self.logger.info(f"BPM 정보 전송: Activation={activation}, ButtonDone={button_press_done}, "
                           f"ButtonStatus={button_status}, Ready={ready}")
            
        except Exception as e:
            self.logger.error(f"BPM 정보 전송 실패: {e}")
    
    def parse_delivery_info(self, data: bytes) -> Optional[Dict]:
        """DELIVERY_INFO UDP 메시지 파싱"""
        try:
            # DELIVERY_INFO 프로토콜 (Message ID: 100, Length: 16)
            if len(data) != 16:
                return None
                
            # 메시지 언패킹 (4s2H8B 형식)
            sdata = struct.unpack('<4s2H8B', data)
            
            start_byte = sdata[0].decode()
            message_id = sdata[1]
            length = sdata[2]
            
            # DELIVERY_INFO 메시지 확인
            if start_byte == 'POLA' and message_id == 100:
                delivery_info = {
                    'start_to_deliver': sdata[3],
                    'current_floor': sdata[4],
                    'target_floor': sdata[5],
                    'robot_location': sdata[6]
                }
                self.logger.info(f"DELIVERY_INFO 수신: {delivery_info}")
                return delivery_info
                
        except Exception as e:
            self.logger.warning(f"DELIVERY_INFO 파싱 실패: {e}")
            
        return None
    
    def check_udp_commands(self):
        """UDP 명령 확인"""
        try:
            data, addr = self.udp_receiver.recvfrom(1024)
            delivery_info = self.parse_delivery_info(data)
            
            if delivery_info and delivery_info['start_to_deliver'] == 1:
                # 배송 시작 명령 수신
                self.current_floor = delivery_info['current_floor']
                self.target_floor = delivery_info['target_floor']
                robot_location = delivery_info['robot_location']
                
                self.logger.info(f"배송 명령 수신: {self.current_floor}층 → {self.target_floor}층, 위치: {robot_location}")
                
                # robot_location에 따라 다른 버튼 감지
                if robot_location == 0:  # 엘리베이터 밖
                    self.logger.info("엘리베이터 밖: up/down 버튼 감지 시작")
                    self.run_elevator_direction_detection()
                elif robot_location == 1:  # 엘리베이터 안
                    self.logger.info("엘리베이터 안: 숫자 버튼 감지 시작")
                    self.run_target_floor_detection()
                
        except socket.timeout:
            pass  # 타임아웃은 정상
        except Exception as e:
            self.logger.error(f"UDP 명령 확인 오류: {e}")
    
    def detect_direction_button(self, target_class_id: int) -> Optional[Tuple[float, float, float]]:
        """지정된 클래스 ID의 버튼을 감지하고 3D 좌표 반환"""
        if not self.is_delivery_active:
            return None
            
        # 프레임 가져오기
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            return None
        
        # 이미지 변환
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # 카메라 매트릭스 가져오기
        K_color, K_depth, dist_coeffs_color, dist_coeffs_depth = self.get_camera_matrices()
        
        # Color 이미지 왜곡 보정
        color_undistorted = self.undistort_image(color_image, K_color, dist_coeffs_color)
        
        # YOLO inference
        results = self.model(color_undistorted, verbose=False)
        
        # 해당 클래스의 detection 찾기
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = box.conf[0].cpu().numpy()
                    
                    if class_id == target_class_id and confidence > 0.5:  # 신뢰도 50% 이상
                        # Bounding box 좌표
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        
                        # 3D 좌표 계산
                        coords_3d = self.get_3d_coordinates(
                            (x1, y1, x2, y2), depth_image, K_depth, dist_coeffs_depth
                        )
                        
                        if coords_3d is not None:
                            class_name = self.class_names[class_id]
                            self.logger.info(f"버튼 감지: {class_name} (신뢰도: {confidence:.2f})")
                            self.logger.info(f"3D 좌표: ({coords_3d[0]:.1f}, {coords_3d[1]:.1f}, {coords_3d[2]:.1f})mm")
                            return coords_3d
        
        return None
    
    def run_elevator_direction_detection(self):
        """엘리베이터 밖에서 up/down 버튼 감지"""
        self.is_delivery_active = True
        self.logger.info("up/down 버튼 감지 시작...")
        
        # up 또는 down 버튼 감지
        target_class_id = None
        if self.target_floor > self.current_floor:
            target_class_id = 10  # 'up'
            button_name = "up"
        elif self.target_floor < self.current_floor:
            target_class_id = 11  # 'down'
            button_name = "down"
        else:
            self.logger.warning("현재층과 목표층이 같습니다.")
            return
        
        max_attempts = 30  # 최대 30번 시도 (약 3초)
        for attempt in range(max_attempts):
            coords_3d = self.detect_direction_button(target_class_id)
            
            if coords_3d is not None:
                # 버튼 위치 전송
                self.send_button_position(coords_3d, coords_3d)
                
                # BPM 활성화 신호 전송
                self.send_bpm_info(activation=1, ready=1)
                
                self.logger.info(f"✅ {button_name} 버튼 감지 및 좌표 전송 완료!")
                break
            else:
                time.sleep(0.1)  # 100ms 대기
        else:
            self.logger.warning(f"❌ {button_name} 버튼을 찾을 수 없습니다.")
        
        self.is_delivery_active = False
        self.logger.info("엘리베이터 방향 감지 완료")
    
    def run_target_floor_detection(self):
        """엘리베이터 안에서 숫자 버튼 감지"""
        self.is_delivery_active = True
        self.logger.info("숫자 버튼 감지 시작...")
        
        # 목표층 숫자 버튼 감지
        if 0 <= self.target_floor <= 9:
            target_class_id = self.target_floor
            button_name = str(self.target_floor)
        else:
            self.logger.warning(f"지원하지 않는 층수: {self.target_floor}")
            return
        
        max_attempts = 30  # 최대 30번 시도 (약 3초)
        for attempt in range(max_attempts):
            coords_3d = self.detect_direction_button(target_class_id)
            
            if coords_3d is not None:
                # 버튼 위치 전송
                self.send_button_position(coords_3d, coords_3d)
                
                # BPM 활성화 신호 전송
                self.send_bpm_info(activation=1, ready=1)
                
                self.logger.info(f"✅ {button_name} 버튼 감지 및 좌표 전송 완료!")
                break
            else:
                time.sleep(0.1)  # 100ms 대기
        else:
            self.logger.warning(f"❌ {button_name} 버튼을 찾을 수 없습니다.")
        
        self.is_delivery_active = False
        self.logger.info("숫자 버튼 감지 완료")
    
    def run_udp_mode(self):
        """UDP 통신 모드 실행"""
        if not self.start_camera():
            return
        
        self.logger.info("🎥 엘리베이터 버튼 감지기 시작!")
        self.logger.info("UDP 포트 5001에서 DELIVERY_INFO 대기 중...")
        self.logger.info("HMI에서 배송 명령을 보내면 자동으로 버튼 감지 및 3D 좌표 전송을 시작합니다.")
        
        try:
            while True:
                # UDP 명령 확인
                self.check_udp_commands()
                
                # 짧은 대기 (CPU 사용률 조절)
                time.sleep(0.01)
                    
        except KeyboardInterrupt:
            self.logger.info("사용자에 의해 중단되었습니다.")
        
        finally:
            self.pipeline.stop()
            self.udp_receiver.close()
            self.logger.info("✅ 프로그램이 종료되었습니다.")

def main():
    """메인 함수"""
    print("엘리베이터 버튼 감지 및 3D 좌표 전송 시스템")
    print("=" * 50)
    
    try:
        detector = ElevatorButtonDetector()
        detector.run_udp_mode()
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()
