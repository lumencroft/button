#!/usr/bin/env python3
"""
Webcam Recorder for NVIDIA Jetson
Records webcam video with OpenCV, saves as MP4
Press 's' to start/stop recording, ESC to exit
"""

import cv2
import numpy as np
import os
from datetime import datetime
import time

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
    
    def run(self):
        """Main recording loop"""
        if not self.initialize_camera():
            return
        
        print("\n🎥 Webcam Recorder Started!")
        print("Controls:")
        print("   's' key: Start/Stop recording")
        print("   ESC key: Exit program")
        print("-" * 40)
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Failed to read frame from camera")
                    break
                
                # Add recording indicator
                if self.recording:
                    # Red circle indicator
                    cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(frame, "REC", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Write frame to video file
                    if self.writer:
                        self.writer.write(frame)
                
                # Add frame counter and FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # Display info on frame
                info_text = f"FPS: {current_fps:.1f} | Frames: {frame_count}"
                cv2.putText(frame, info_text, (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, info_text, (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                
                # Display frame
                cv2.imshow('Webcam Recorder', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC key
                    break
                elif key == ord('s') or key == ord('S'):  # 's' key
                    if self.recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
                
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        
        finally:
            # Cleanup
            if self.recording:
                self.stop_recording()
            
            if self.cap:
                self.cap.release()
            
            cv2.destroyAllWindows()
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

if __name__ == "__main__":
    main()
