#!/usr/bin/env python3
"""
RealSense Camera + YOLO + 3D Coordinate Calculation
Receives RGB/Depth streams from RealSense camera, performs YOLO inference,
and calculates 3D coordinates of bounding boxes considering distortion correction.
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
        """Initialize RealSense + YOLO + 3D coordinate calculation class"""
        
        # Load YOLO model
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        
        # RealSense pipeline setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Stream configuration
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Camera intrinsic parameters (previously extracted values)
        self.color_intrinsics = None
        self.depth_intrinsics = None
        self.depth_scale = None
        
        # Class names
        self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'up', 'down']
        
        # Initialize coordinate calculator
        self.coord_calculator = None
        
        # UDP transmission setup
        self.udp_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.target_ip = "192.168.1.131"
        self.target_port = 5003  # Use same port for transmission
        
        # UDP reception setup (receive ready signal)
        self.udp_receiver = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        self.receiver_ip = "0.0.0.0"  # Receive from all interfaces
        self.receiver_port = 5003
        self.udp_receiver.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allow port reuse
        self.udp_receiver.bind((self.receiver_ip, self.receiver_port))
        
        # BUTTON_POSITION protocol setup (referencing psock structure)
        self.start_byte = 'POLA'
        self.message_id = 102  # BUTTON_POSITION
        self.length = 32  # Length field value
        
        # State management
        self.state = "waiting_ready"  # waiting_ready, looking_up, looking_7
        self.ready_signal_received = False
        self.up_button_sent = False
        self.button_7_sent = False
        self.current_sequence = "up"  # "up" or "7" - which button to look for next
        
        # Timer for button detection transmission
        self.last_send_time = 0
        self.send_interval = 1.0  # Send every 1 second
        
        # Thread control
        self.running = True
        self.udp_thread = None
        

        
    def start_camera(self):
        """Start RealSense camera"""
        try:
            # Start pipeline
            profile = self.pipeline.start(self.config)
            
            # Get intrinsic parameters
            color_stream = profile.get_stream(rs.stream.color)
            self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
            depth_stream = profile.get_stream(rs.stream.depth)
            self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
            
            print("✅ RealSense camera connected successfully!")
            print(f"Color resolution: {self.color_intrinsics.width}x{self.color_intrinsics.height}")
            print(f"Depth resolution: {self.depth_intrinsics.width}x{self.depth_intrinsics.height}")
            
            # Check depth unit
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            print(f"Depth unit scale: {self.depth_scale}")
            print(f"Actual depth value = raw_depth * {self.depth_scale} (in meters)")
            
            # Initialize coordinate calculator
            self.coord_calculator = CoordinateCalculator(self.depth_scale)
            
            return True
            
        except Exception as e:
            print(f"❌ RealSense camera connection failed: {e}")
            return False
    
    def udp_receiver_thread(self):
        """UDP reception thread - receive ready signal and change state"""
        print(f"📡 UDP reception started: {self.receiver_ip}:{self.receiver_port}")
        
        while self.running:
            try:
                # Receive BPM_INFO protocol (16 bytes)
                data, addr = self.udp_receiver.recvfrom(16)
                
                if len(data) == 16:
                    # Parse protocol
                    unpacked = struct.unpack("<4s2H8B", data)
                    start_byte = unpacked[0].decode()
                    message_id = unpacked[1]
                    length = unpacked[2]
                    activation = unpacked[3]
                    button_press_done = unpacked[4]
                    button_status = unpacked[5]
                    ready = unpacked[6]
                    operate_tray_door = unpacked[7]
                    
                    # If BPM_INFO message and ready signal is 1
                    if start_byte == 'POLA' and message_id == 103 and ready == 1:
                        print(f"🎯 Ready signal received! (from {addr[0]}:{addr[1]})")
                        self.ready_signal_received = True
                        
                        # Determine which button to look for based on current sequence
                        if self.current_sequence == "up":
                            self.state = "looking_up"
                            self.up_button_sent = False
                            print("🎯 Looking for UP button...")
                        elif self.current_sequence == "7":
                            self.state = "looking_7"
                            self.button_7_sent = False
                            print("🎯 Looking for 7 button...")
                        
            except Exception as e:
                if self.running:  # Only print error if not normal termination
                    print(f"❌ UDP reception error: {e}")
                break
    
    def start_udp_receiver(self):
        """Start UDP reception thread"""
        self.udp_thread = threading.Thread(target=self.udp_receiver_thread, daemon=True)
        self.udp_thread.start()
    
    def send_button_position(self, x_3d, y_3d, z_3d, button_type="up"):
        """Send button position via UDP"""
        try:
            # Current time
            current_time = time.time()
            
            # Create packet according to BUTTON_POSITION protocol (total = 32 bytes)
            # Header: start_byte(4) + message_id(2) + length(2) = 8 bytes
            # Payload: time(8) + button_pos[3](12) + type(1) + reserved[3](3) = 24 bytes
            # Total: 32 bytes (header 8 + payload 24)
            
            message = struct.pack(
                "<4s2Hd3fB3B",  # Format: 4s(start_byte) + 2H(message_id, length) + d(time) + 3f(button_pos) + B(type) + 3B(reserved) = 32 bytes
                self.start_byte.encode(),  # start_byte: 'POLA'
                self.message_id,           # message_id: 102
                32,                        # length: 32 (fixed)
                current_time,              # time: double
                float(x_3d),               # button_pos[0]: float
                float(y_3d),               # button_pos[1]: float  
                float(z_3d),               # button_pos[2]: float
                1,                         # type: 1 (camera coordinate)
                0, 0, 0                    # reserved[3]: 0, 0, 0
            )
            
            # UDP transmission
            self.udp_socket.sendto(message, (self.target_ip, self.target_port))
            print(f"📡 UDP transmission completed: {button_type.upper()} button position ({x_3d:.1f}, {y_3d:.1f}, {z_3d:.1f})mm -> {self.target_ip}:{self.target_port}")
            
        except Exception as e:
            print(f"❌ UDP transmission failed: {e}")
    
    def run(self):
        """Main execution loop"""
        if not self.start_camera():
            return
        
        # Start UDP reception thread
        self.start_udp_receiver()
        
        # Get camera matrices
        K_color, K_depth, dist_coeffs_color, dist_coeffs_depth = self.coord_calculator.get_camera_matrices(
            self.color_intrinsics, self.depth_intrinsics
        )
        
        print("\n🎥 RealSense + YOLO + 3D coordinate calculation started!")
        print("🎯 Waiting for ready signal... (receiving on UDP port 5003)")
        print("🎯 Sequence: Ready signal → UP button → Ready signal → 7 button → Ready signal → UP button...")
        print(f"📡 Transmission target: {self.target_ip}:{self.target_port} (sending button positions)")
        print("\n⌨️  Keyboard input:")
        print("   'c' key: distortion correction toggle")
        print("   'm' key: 3D calculation matrix toggle (Color/Depth)")
        print("   'r' key: reset state")
        print("   ESC key: exit")
        print("-" * 50)
        
        try:
            while True:
                # Get frames
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                

                
                # Convert images
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Color image distortion correction
                color_undistorted = self.coord_calculator.undistort_image(color_image, K_color, dist_coeffs_color)
                
                # YOLO inference
                results = self.model(color_undistorted, verbose=False)
                
                # Result visualization
                annotated_image = color_undistorted.copy()
                
                # Determine target button based on state
                target_class_id = None
                target_class_name = None
                
                if self.state == "looking_up":
                    target_class_id = 10  # 'up' button
                    target_class_name = "up"
                elif self.state == "looking_7":
                    target_class_id = 7   # '7' button
                    target_class_name = "7"
                
                # Calculate 3D coordinates for each detection
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            confidence = box.conf[0].cpu().numpy()
                            if confidence < 0.5:
                                continue
                            class_id = int(box.cls[0].cpu().numpy())
                            
                            # Get class name
                            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                            
                            # Calculate 3D coordinates
                            x_3d, y_3d, z_3d = self.coord_calculator.get_3d_coordinates(
                                (x1, y1, x2, y2), depth_image, K_depth, dist_coeffs_depth, K_color, dist_coeffs_color
                            )
                            
                            # Draw bounding box
                            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Text information
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                            
                            if x_3d is not None:
                                # Debug information for depth value verification
                                center_depth_raw = depth_image[center_y, center_x]
                                center_depth_meters = center_depth_raw * self.depth_scale if self.depth_scale else center_depth_raw
                                
                                text = f"{class_name}: {confidence:.2f}"
                                text_3d = f"3D: ({x_3d:.1f}, {y_3d:.1f}, {z_3d:.1f})mm"
                                
                                cv2.putText(annotated_image, text, (x1, y1-30), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                cv2.putText(annotated_image, text_3d, (x1, y1-15), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                                
                                # Draw circle at center point
                                cv2.circle(annotated_image, (center_x, center_y), 3, (0, 0, 255), -1)
                                
                                # Send UDP when target button is detected
                                if (target_class_id is not None and class_id == target_class_id and 
                                    not (self.state == "looking_up" and self.up_button_sent) and
                                    not (self.state == "looking_7" and self.button_7_sent)):
                                    
                                    current_time = time.time()
                                    if current_time - self.last_send_time >= self.send_interval:
                                        self.send_button_position(x_3d, y_3d, z_3d, target_class_name)
                                        self.last_send_time = current_time
                                        
                                        if self.state == "looking_up":
                                            self.up_button_sent = True
                                            self.state = "waiting_ready"
                                            self.current_sequence = "7"  # Next time ready signal comes, look for 7 button
                                            print("🎯 UP button transmission completed! Waiting for next ready signal to look for 7 button...")
                                        elif self.state == "looking_7":
                                            self.button_7_sent = True
                                            self.state = "waiting_ready"
                                            self.current_sequence = "up"  # Next time ready signal comes, look for up button
                                            print("🎯 7 button transmission completed! Waiting for next ready signal to look for UP button...")
                            else:
                                text = f"{class_name}: {confidence:.2f} (No depth)"
                                cv2.putText(annotated_image, text, (x1, y1-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # Depth image visualization (colormap)
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
                )
                
                # Display current state information on screen
                if self.state == "waiting_ready":
                    next_button = "UP" if self.current_sequence == "up" else "7"
                    status_info = f"State: Waiting for ready signal... (Next: {next_button} button)"
                    color = (0, 255, 255)  # Yellow
                elif self.state == "looking_up":
                    status_info = "State: Looking for UP button..."
                    color = (0, 255, 0)    # Green
                elif self.state == "looking_7":
                    status_info = "State: Looking for 7 button..."
                    color = (255, 0, 0)    # Blue
                else:
                    status_info = f"State: {self.state}"
                    color = (255, 255, 255)  # White
                
                cv2.putText(annotated_image, status_info, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)  # Black outline
                cv2.putText(annotated_image, status_info, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Display distortion correction status
                distortion_info = f"Distortion: {'ON' if self.coord_calculator.use_undistortion else 'OFF'}"
                cv2.putText(annotated_image, distortion_info, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_image, distortion_info, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.coord_calculator.use_undistortion else (0, 0, 255), 1)
                
                # Display 3D calculation matrix status
                matrix_info = f"3D Matrix: {'Color' if self.coord_calculator.use_color_matrix_for_3d else 'Depth'}"
                cv2.putText(annotated_image, matrix_info, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_image, matrix_info, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0) if self.coord_calculator.use_color_matrix_for_3d else (255, 165, 0), 1)
                
                # Display on screen
                cv2.imshow('RealSense + YOLO + 3D', annotated_image)
                cv2.imshow('Depth', depth_colormap)
                
                # Keyboard input handling
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    break
                elif key == ord('c'):  # 'c' key: distortion correction toggle
                    self.coord_calculator.use_undistortion = not self.coord_calculator.use_undistortion
                    status = "ON" if self.coord_calculator.use_undistortion else "OFF"
                    print(f"\n🔧 Distortion correction toggle: {status}")
                elif key == ord('m'):  # 'm' key: 3D calculation matrix toggle
                    self.coord_calculator.use_color_matrix_for_3d = not self.coord_calculator.use_color_matrix_for_3d
                    status = "Color" if self.coord_calculator.use_color_matrix_for_3d else "Depth"
                    print(f"\n🎯 3D calculation matrix toggle: {status}")
                elif key == ord('r'):  # 'r' key: state reset
                    self.state = "waiting_ready"
                    self.up_button_sent = False
                    self.button_7_sent = False
                    self.current_sequence = "up"  # Reset to start with UP button
                    print(f"\n🔄 State reset: Waiting for ready signal... (Next: UP button)")
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        
        finally:
            # Cleanup
            self.running = False
            self.pipeline.stop()
            self.udp_socket.close()
            self.udp_receiver.close()
            cv2.destroyAllWindows()
            print("✅ Program terminated.")

def main():
    """Main function"""
    print("RealSense + YOLO + 3D Coordinate Calculation Program")
    print("=" * 50)
    
    # Check model path
    model_path = "runs/train/clean_training2/weights/best.pt"
    
    try:
        app = RealSenseYOLO3D(model_path)
        app.run()
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Please check the following:")
        print("1. RealSense camera is connected")
        print("2. YOLO model file exists")
        print("3. Required libraries are installed")

if __name__ == "__main__":
    main()
