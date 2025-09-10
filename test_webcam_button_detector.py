#!/usr/bin/env python3
"""
웹캠을 사용한 엘리베이터 버튼 감지 테스트 코드
RealSense 대신 일반 웹캠을 사용하여 버튼 감지 기능을 테스트합니다.
"""

import numpy as np
import cv2
import socket
import struct
import time
import logging
import os
import sys
from ultralytics import YOLO
from typing import Optional, Tuple, Dict, Any

# psock 모듈 import
home_path = os.path.expanduser("~")
sys.path.append(f"{home_path}/ws")
from psock.udp.udp_tx import UdpTx

class WebcamButtonDetector:
    def __init__(self, model_path="yolo11n.pt"):
        """웹캠 버튼 감지기 초기화"""
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # YOLO 모델 로드
        self.logger.info(f"YOLO 모델 로딩: {model_path}")
        self.model = YOLO(model_path)
        
        # 클래스 이름 (숫자 0-9, up, down)
        self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'up', 'down']
        
        # 웹캠 설정
        self.cap = None
        self.camera_width = 640
        self.camera_height = 480
        
        # UDP 통신 설정 (psock 모듈 사용)
        self.udp_tx = UdpTx()
        self.setup_udp()
        
        # 상태 변수
        self.current_floor = 1  # 현재 층수
        self.target_floor = 1   # 목적지 층수
        self.is_delivery_active = False
        
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
        """웹캠 시작"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.logger.error("❌ 웹캠을 열 수 없습니다.")
                return False
            
            # 웹캠 설정
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # 실제 설정된 값 확인
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            self.logger.info("✅ 웹캠 연결 성공!")
            self.logger.info(f"해상도: {actual_width}x{actual_height}")
            self.logger.info(f"FPS: {actual_fps}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 웹캠 연결 실패: {e}")
            return False
    
    def get_3d_coordinates(self, bbox: Tuple[int, int, int, int], depth_image: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Bounding box의 3D 좌표 계산 (웹캠용 가상 depth)"""
        x1, y1, x2, y2 = bbox
        
        # Bounding box 중심점
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        # 가상 depth 값 생성 (웹캠용)
        # 실제로는 depth 카메라가 없으므로 고정값 사용
        depth_mm = 1000.0  # 1미터 고정
        
        # 기본 카메라 매트릭스 사용 (640x480 기준)
        fx = 640.0  # 기본 초점거리
        fy = 480.0
        cx = 320.0  # 기본 주점
        cy = 240.0
        
        # 3D 좌표 계산 (mm 단위)
        x_3d = (center_x - cx) * depth_mm / fx
        y_3d = (center_y - cy) * depth_mm / fy
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
            
        # 웹캠에서 프레임 가져오기
        ret, color_image = self.cap.read()
        if not ret:
            return None
        
        # 가상 depth 이미지 생성 (웹캠용)
        depth_image = np.random.randint(1000, 3000, (self.camera_height, self.camera_width), dtype=np.uint16)
        
        # YOLO inference
        results = self.model(color_image, verbose=False)
        
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
                            (x1, y1, x2, y2), depth_image
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
    
    def wait_for_7_signal(self):
        """UDP에서 7번 신호를 기다리고 기존 로직으로 진행"""
        self.logger.info("🔢 UDP에서 7번 신호 대기 중...")
        self.logger.info("HMI에서 target_floor=7인 DELIVERY_INFO를 보내주세요.")
        
        # 7번 신호를 기다리는 상태로 설정
        self.target_floor = 7
        
        # UDP 명령 확인 루프 (무한정 대기, 키보드 입력도 처리)
        while True:
            # 키보드 입력 확인
            import msvcrt
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8').lower()
                
                if key == 'u':
                    self.logger.info("⬆️ up 버튼 감지 시작!")
                    self.run_up_button_detection()
                    return
                elif key == 'q':
                    self.logger.info("프로그램을 종료합니다.")
                    return
                else:
                    self.logger.info(f"알 수 없는 키: {key}")
            
            # UDP 명령 확인
            try:
                data, addr = self.udp_receiver.recvfrom(1024)
                delivery_info = self.parse_delivery_info(data)
                
                if delivery_info and delivery_info['start_to_deliver'] == 1:
                    # 배송 시작 명령 수신
                    self.current_floor = delivery_info['current_floor']
                    self.target_floor = delivery_info['target_floor']
                    robot_location = delivery_info['robot_location']
                    
                    self.logger.info(f"배송 명령 수신: {self.current_floor}층 → {self.target_floor}층, 위치: {robot_location}")
                    
                    # 7번 신호가 왔으면 기존 로직으로 진행
                    if self.target_floor == 7:
                        self.logger.info("✅ 7번 신호 수신! 기존 로직으로 진행...")
                        self.run_target_floor_detection()
                        return
                    else:
                        self.logger.info(f"다른 층수 신호 수신: {self.target_floor}층")
                        
            except socket.timeout:
                pass  # 타임아웃은 정상
            except Exception as e:
                self.logger.error(f"UDP 명령 확인 오류: {e}")
            
            time.sleep(0.01)
    
    def run_up_button_detection(self):
        """up 버튼 전용 감지 로직"""
        self.is_delivery_active = True
        self.logger.info("⬆️ up 버튼 전용 감지 시작...")
        self.logger.info("키보드 단축키: '7' 키로 7번 신호 대기 모드로 전환 가능")
        
        target_class_id = 10  # 'up' 버튼
        button_name = "up"
        
        max_attempts = 30  # 최대 30번 시도 (약 3초)
        for attempt in range(max_attempts):
            # 키보드 입력 확인
            import msvcrt
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8').lower()
                
                if key == '7':
                    self.logger.info("🔢 7번 신호 대기 모드로 전환!")
                    self.is_delivery_active = False
                    self.wait_for_7_signal()
                    return
                elif key == 'q':
                    self.logger.info("프로그램을 종료합니다.")
                    self.is_delivery_active = False
                    return
            
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
        self.logger.info("up 버튼 감지 완료")
    
    def run_keyboard_mode(self):
        """키보드 입력 모드 실행"""
        if not self.start_camera():
            return
        
        self.logger.info("🎥 웹캠 버튼 감지기 시작! (키보드 모드)")
        self.logger.info("키보드 단축키:")
        self.logger.info("  'u' 키: up 버튼 감지 시작")
        self.logger.info("  '7' 키: 7번 신호 대기 (UDP에서 target_floor=7 신호 대기)")
        self.logger.info("  'q' 키: 프로그램 종료")
        
        try:
            while True:
                # 키보드 입력 확인
                import msvcrt
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8').lower()
                    
                    if key == 'u':
                        self.logger.info("⬆️ up 버튼 감지 시작!")
                        self.run_up_button_detection()
                    elif key == '7':
                        self.logger.info("🔢 7번 버튼 신호 대기 중...")
                        self.wait_for_7_signal()
                    elif key == 'q':
                        self.logger.info("프로그램을 종료합니다.")
                        break
                    else:
                        self.logger.info(f"알 수 없는 키: {key}")
                
                # 짧은 대기 (CPU 사용률 조절)
                time.sleep(0.01)
                    
        except KeyboardInterrupt:
            self.logger.info("사용자에 의해 중단되었습니다.")
        
        finally:
            if self.cap:
                self.cap.release()
            self.udp_receiver.close()
            cv2.destroyAllWindows()
            self.logger.info("✅ 프로그램이 종료되었습니다.")
    
    def run_udp_mode(self):
        """UDP 통신 모드 실행"""
        if not self.start_camera():
            return
        
        self.logger.info("🎥 웹캠 버튼 감지기 시작!")
        self.logger.info("UDP 포트 5001에서 DELIVERY_INFO 대기 중...")
        self.logger.info("HMI에서 배송 명령을 보내면 자동으로 버튼 감지 및 3D 좌표 전송을 시작합니다.")
        self.logger.info("키보드 단축키:")
        self.logger.info("  'u' 키: up 버튼 감지 시작")
        self.logger.info("  '7' 키: 7번 버튼 감지 시작")
        self.logger.info("  'q' 키: 프로그램 종료")
        
        try:
            while True:
                # UDP 명령 확인
                self.check_udp_commands()
                
                # 짧은 대기 (CPU 사용률 조절)
                time.sleep(0.01)
                    
        except KeyboardInterrupt:
            self.logger.info("사용자에 의해 중단되었습니다.")
        
        finally:
            if self.cap:
                self.cap.release()
            self.udp_receiver.close()
            self.logger.info("✅ 프로그램이 종료되었습니다.")

def main():
    """메인 함수"""
    print("웹캠 엘리베이터 버튼 감지 및 3D 좌표 전송 시스템")
    print("=" * 60)
    print("모드를 선택하세요:")
    print("1. UDP 모드 (HMI 통신)")
    print("2. 키보드 모드 (u: up 버튼, 7: 7번 신호 대기)")
    
    try:
        mode = input("모드 선택 (1 또는 2): ").strip()
        detector = WebcamButtonDetector()
        
        if mode == "1":
            detector.run_udp_mode()
        elif mode == "2":
            detector.run_keyboard_mode()
        else:
            print("잘못된 선택입니다. 키보드 모드로 시작합니다.")
            detector.run_keyboard_mode()
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    main()
