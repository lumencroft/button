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
        
        # Set default settings
        self.show_depth = False  # Depth visualization off by default
        self.use_undistortion = False  # Distortion correction off by default
        
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
        
        # Frame collection for averaging
        self.frame_collection_mode = False
        self.collected_frames = []  # Store 5 frames of 3D coordinates
        self.target_frames = 5  # Collect 5 frames
        self.median_frames = 3  # Use median 3 frames for averaging
        
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
            
            # Set default settings
            self.coord_calculator.use_undistortion = self.use_undistortion
            
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
                        
                        # Wait 0.5 seconds before starting frame collection
                        print("⏳ Waiting 0.5 seconds before starting frame collection...")
                        time.sleep(2)
                        
                        # Start frame collection mode
                        self.frame_collection_mode = True
                        self.collected_frames = []
                        
                        # Determine which button to look for based on current sequence
                        if self.current_sequence == "up":
                            self.state = "looking_up"
                            self.up_button_sent = False
                            print("🎯 Collecting 5 frames for UP button...")
                        elif self.current_sequence == "7":
                            self.state = "looking_7"
                            self.button_7_sent = False
                            print("🎯 Collecting 5 frames for 7 button...")
                        
            except Exception as e:
                if self.running:  # Only print error if not normal termination
                    print(f"❌ UDP reception error: {e}")
                break
    
    def start_udp_receiver(self):
        """Start UDP reception thread"""
        self.udp_thread = threading.Thread(target=self.udp_receiver_thread, daemon=True)
        self.udp_thread.start()
    
    def calculate_averaged_position(self, positions):
        """Calculate averaged position from collected frames using median 3 frames"""
        if len(positions) < self.median_frames:
            return None
        
        # Sort positions by each coordinate
        x_coords = sorted([pos[0] for pos in positions])
        y_coords = sorted([pos[1] for pos in positions])
        z_coords = sorted([pos[2] for pos in positions])
        
        # Get median values
        median_x = x_coords[len(x_coords) // 2]
        median_y = y_coords[len(y_coords) // 2]
        median_z = z_coords[len(z_coords) // 2]
        
        # Calculate average of median 3 frames
        start_idx = len(positions) // 2 - 1
        end_idx = start_idx + self.median_frames
        
        if end_idx > len(positions):
            start_idx = len(positions) - self.median_frames
            end_idx = len(positions)
        
        median_positions = positions[start_idx:end_idx]
        
        avg_x = sum(pos[0] for pos in median_positions) / len(median_positions)
        avg_y = sum(pos[1] for pos in median_positions) / len(median_positions)
        avg_z = sum(pos[2] for pos in median_positions) / len(median_positions)
        
        return avg_x, avg_y, avg_z
    
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
        print(f"🔧 Settings: Depth visualization OFF, Distortion correction OFF")
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
                
                # Color image distortion correction (only if enabled)
                if self.coord_calculator.use_undistortion:
                    color_undistorted = self.coord_calculator.undistort_image(color_image, K_color, dist_coeffs_color)
                else:
                    color_undistorted = color_image
                
                # YOLO inference
                results = self.model(color_undistorted, verbose=False)
                
                # Result visualization (only if needed for processing)
                # Note: No display windows will be shown
                
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
                            
                            # Calculate center point for 3D coordinate calculation
                            center_x = (x1 + x2) // 2
                            center_y = (y1 + y2) // 2
                                
                                # Handle frame collection for target button
                                if (target_class_id is not None and class_id == target_class_id and 
                                    self.frame_collection_mode and
                                    not (self.state == "looking_up" and self.up_button_sent) and
                                    not (self.state == "looking_7" and self.button_7_sent)):
                                    
                                    # Collect frame data
                                    self.collected_frames.append((x_3d, y_3d, z_3d))
                                    print(f"📊 Frame {len(self.collected_frames)}/{self.target_frames} collected: ({x_3d:.1f}, {y_3d:.1f}, {z_3d:.1f})mm")
                                    
                                    # Check if we have enough frames
                                    if len(self.collected_frames) >= self.target_frames:
                                        # Calculate averaged position
                                        avg_pos = self.calculate_averaged_position(self.collected_frames)
                                        
                                        if avg_pos is not None:
                                            avg_x, avg_y, avg_z = avg_pos
                                            print(f"📊 Averaged position from {len(self.collected_frames)} frames: ({avg_x:.1f}, {avg_y:.1f}, {avg_z:.1f})mm")
                                            
                                            # Send averaged position
                                            self.send_button_position(avg_x, avg_y, avg_z, target_class_name)
                                            
                                            # Update state
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
                                        
                                        # Reset frame collection
                                        self.frame_collection_mode = False
                                        self.collected_frames = []
                            else:
                                # No depth data available
                                pass
                
                # Print current state information to console
                if self.state == "waiting_ready":
                    next_button = "UP" if self.current_sequence == "up" else "7"
                    print(f"State: Waiting for ready signal... (Next: {next_button} button)")
                elif self.state == "looking_up":
                    if self.frame_collection_mode:
                        print(f"State: Collecting UP button frames... ({len(self.collected_frames)}/{self.target_frames})")
                    else:
                        print("State: Looking for UP button...")
                elif self.state == "looking_7":
                    if self.frame_collection_mode:
                        print(f"State: Collecting 7 button frames... ({len(self.collected_frames)}/{self.target_frames})")
                    else:
                        print("State: Looking for 7 button...")
                else:
                    print(f"State: {self.state}")
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        
        finally:
            # Cleanup
            self.running = False
            self.pipeline.stop()
            self.udp_socket.close()
            self.udp_receiver.close()
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
