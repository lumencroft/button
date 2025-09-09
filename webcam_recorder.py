#!/usr/bin/env python3
"""
Webcam Recorder for NVIDIA Jetson
Records webcam video with OpenCV, saves as MP4
Command line interface for recording control
"""

import cv2
import numpy as np
import os
from datetime import datetime
import time
import threading
import sys

class WebcamRecorder:
    def __init__(self, camera_index=0, fps=30, resolution=(640, 480)):
        """
        Initialize webcam recorder
        
        Args:
            camera_index (int): Camera index (0 for default camera)
            fps (int): Frames per second
            resolution (tuple): Video resolution (width, height)
        """
        self.camera_index = camera_index
        self.fps = fps
        self.resolution = resolution
        self.cap = None
        self.writer = None
        self.recording = False
        self.output_filename = None
        self.running = True
        self.frame_count = 0
        self.start_time = None
        
        # Create output directory if it doesn't exist
        self.output_dir = "recordings"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def initialize_camera(self):
        """Initialize camera capture"""
        try:
            # For NVIDIA Jetson, use GStreamer backend for better performance
            # This pipeline is optimized for Jetson hardware
            gst_pipeline = (
                f"nvarguscamerasrc sensor-id={self.camera_index} ! "
                f"video/x-raw(memory:NVMM), width={self.resolution[0]}, height={self.resolution[1]}, "
                f"framerate={self.fps}/1, format=NV12 ! "
                f"nvvidconv flip-method=0 ! "
                f"video/x-raw, width={self.resolution[0]}, height={self.resolution[1]}, format=BGRx ! "
                f"videoconvert ! video/x-raw, format=BGR ! appsink"
            )
            
            # Try Jetson-optimized pipeline first
            self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            
            if not self.cap.isOpened():
                print("⚠️  Jetson pipeline failed, trying standard camera...")
                # Fallback to standard camera
                self.cap = cv2.VideoCapture(self.camera_index)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            if not self.cap.isOpened():
                raise Exception("Failed to open camera")
            
            print(f"✅ Camera initialized successfully!")
            print(f"   Resolution: {self.resolution[0]}x{self.resolution[1]}")
            print(f"   FPS: {self.fps}")
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            return False
    
    def start_recording(self):
        """Start video recording"""
        if self.recording:
            print("⚠️  Already recording!")
            return
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_filename = os.path.join(self.output_dir, f"recording_{timestamp}.mp4")
        
        # Get actual camera properties
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # For NVIDIA Jetson, use hardware-accelerated codec
        # H.264 codec is well supported on Jetson
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for better compatibility
        
        self.writer = cv2.VideoWriter(
            self.output_filename,
            fourcc,
            actual_fps,
            (actual_width, actual_height)
        )
        
        if not self.writer.isOpened():
            print("❌ Failed to initialize video writer!")
            return False
        
        self.recording = True
        print(f"🔴 Recording started: {self.output_filename}")
        print(f"   Resolution: {actual_width}x{actual_height}")
        print(f"   FPS: {actual_fps}")
        return True
    
    def stop_recording(self):
        """Stop video recording"""
        if not self.recording:
            print("⚠️  Not currently recording!")
            return
        
        if self.writer:
            self.writer.release()
            self.writer = None
        
        self.recording = False
        print(f"⏹️  Recording stopped: {self.output_filename}")
        
        # Check if file was created and get size
        if os.path.exists(self.output_filename):
            file_size = os.path.getsize(self.output_filename)
            print(f"   File size: {file_size / (1024*1024):.2f} MB")
        else:
            print("⚠️  Warning: Output file not found!")
    
    def camera_loop(self):
        """Camera processing loop (runs in background thread)"""
        if not self.initialize_camera():
            return
        
        print("\n🎥 Webcam Recorder Started!")
        print("Camera is running in background...")
        print("\nCommands:")
        print("   'start' or 's': Start recording")
        print("   'stop' or 'q': Stop recording")
        print("   'status': Show current status")
        print("   'exit': Exit program")
        print("-" * 50)
        
        self.start_time = time.time()
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame from camera")
                    break
                
                # Process frame for recording
                if self.recording and self.writer:
                    self.writer.write(frame)
                
                # Update frame counter and FPS
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                current_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # Print status every 5 seconds
                if self.frame_count % 150 == 0:  # Assuming 30fps, so every 5 seconds
                    status = "RECORDING" if self.recording else "IDLE"
                    print(f"📊 Status: {status} | FPS: {current_fps:.1f} | Frames: {self.frame_count}")
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
                
        except Exception as e:
            print(f"❌ Camera loop error: {e}")
        
        finally:
            # Cleanup
            if self.recording:
                self.stop_recording()
            
            if self.cap:
                self.cap.release()
            
            print("✅ Camera loop terminated")
    
    def command_loop(self):
        """Command line interface loop"""
        while self.running:
            try:
                command = input("\n> ").strip().lower()
                
                if command in ['start', 's']:
                    if not self.recording:
                        self.start_recording()
                    else:
                        print("⚠️  Already recording!")
                
                elif command in ['stop', 'q']:
                    if self.recording:
                        self.stop_recording()
                    else:
                        print("⚠️  Not currently recording!")
                
                elif command == 'status':
                    elapsed_time = time.time() - self.start_time if self.start_time else 0
                    current_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                    status = "RECORDING" if self.recording else "IDLE"
                    print(f"📊 Current Status: {status}")
                    print(f"📊 FPS: {current_fps:.1f}")
                    print(f"📊 Total Frames: {self.frame_count}")
                    if self.recording:
                        print(f"📊 Recording: {self.output_filename}")
                
                elif command in ['exit', 'quit']:
                    print("👋 Exiting program...")
                    self.running = False
                    break
                
                elif command == 'help':
                    print("\nAvailable commands:")
                    print("   start, s  - Start recording")
                    print("   stop, q   - Stop recording")
                    print("   status    - Show current status")
                    print("   help      - Show this help")
                    print("   exit      - Exit program")
                
                else:
                    print("❌ Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n👋 Exiting program...")
                self.running = False
                break
            except EOFError:
                print("\n👋 Exiting program...")
                self.running = False
                break
    
    def run(self):
        """Main execution function"""
        # Start camera loop in background thread
        camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        camera_thread.start()
        
        # Wait a moment for camera to initialize
        time.sleep(2)
        
        # Start command line interface in main thread
        self.command_loop()
        
        # Cleanup
        self.running = False
        print("✅ Program terminated")

def main():
    """Main function"""
    print("Webcam Recorder for NVIDIA Jetson")
    print("=" * 40)
    
    # Configuration for NVIDIA Jetson
    # You can adjust these parameters based on your Jetson model
    config = {
        'camera_index': 0,  # Default camera
        'fps': 30,          # Frames per second
        'resolution': (640, 480)  # Width, Height
    }
    
    # For higher resolution on newer Jetson models, you can use:
    # 'resolution': (1280, 720)  # 720p
    # 'resolution': (1920, 1080) # 1080p (may require more processing power)
    
    try:
        recorder = WebcamRecorder(**config)
        recorder.run()
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Please check the following:")
        print("1. Camera is connected and accessible")
        print("2. OpenCV is properly installed")
        print("3. For Jetson: GStreamer plugins are installed")
    except KeyboardInterrupt:
        print("\n👋 Program interrupted by user")
        sys.exit(0)

if __name__ == "__main__":
    main()
