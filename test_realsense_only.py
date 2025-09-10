#!/usr/bin/env python3
"""
RealSense 카메라만 테스트하는 스크립트
카메라 연결 상태와 기본 스트림을 확인합니다.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time

def test_realsense_connection():
    """RealSense 카메라 연결 테스트"""
    print("🔍 RealSense 카메라 연결 테스트 시작...")
    
    # RealSense 컨텍스트 생성
    ctx = rs.context()
    devices = ctx.query_devices()
    
    print(f"📱 발견된 RealSense 장치 수: {len(devices)}")
    
    if len(devices) == 0:
        print("❌ RealSense 카메라가 연결되지 않았습니다.")
        print("확인사항:")
        print("1. USB 케이블 연결 확인")
        print("2. USB 3.0 포트 사용 확인")
        print("3. Intel RealSense SDK 설치 확인")
        return False
    
    # 첫 번째 장치 정보 출력
    device = devices[0]
    print(f"✅ RealSense 카메라 발견: {device.get_info(rs.camera_info.name)}")
    print(f"   시리얼 번호: {device.get_info(rs.camera_info.serial_number)}")
    print(f"   펌웨어 버전: {device.get_info(rs.camera_info.firmware_version)}")
    
    return True

def test_realsense_streams():
    """RealSense 스트림 테스트"""
    print("\n🎥 RealSense 스트림 테스트 시작...")
    
    # 파이프라인 생성
    pipeline = rs.pipeline()
    config = rs.config()
    
    try:
        # 스트림 설정
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        print("📡 스트림 설정 완료")
        print("   - Color: 640x480, BGR8, 30fps")
        print("   - Depth: 640x480, Z16, 30fps")
        
        # 파이프라인 시작
        print("🚀 파이프라인 시작 중...")
        profile = pipeline.start(config)
        print("✅ 파이프라인 시작 성공!")
        
        # 스트림 정보 출력
        color_stream = profile.get_stream(rs.stream.color)
        depth_stream = profile.get_stream(rs.stream.depth)
        
        color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        
        print(f"📊 Color 스트림 정보:")
        print(f"   - 해상도: {color_intrinsics.width}x{color_intrinsics.height}")
        print(f"   - 초점거리: fx={color_intrinsics.fx:.2f}, fy={color_intrinsics.fy:.2f}")
        print(f"   - 주점: cx={color_intrinsics.ppx:.2f}, cy={color_intrinsics.ppy:.2f}")
        
        print(f"📊 Depth 스트림 정보:")
        print(f"   - 해상도: {depth_intrinsics.width}x{depth_intrinsics.height}")
        print(f"   - 초점거리: fx={depth_intrinsics.fx:.2f}, fy={depth_intrinsics.fy:.2f}")
        print(f"   - 주점: cx={depth_intrinsics.ppx:.2f}, cy={depth_intrinsics.ppy:.2f}")
        
        # Depth 스케일 확인
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"📏 Depth 스케일: {depth_scale} (raw_depth * {depth_scale} = 미터)")
        
        # 5초간 프레임 수신 테스트
        print("\n🎬 5초간 프레임 수신 테스트...")
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 5:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if color_frame and depth_frame:
                frame_count += 1
                if frame_count % 30 == 0:  # 1초마다 출력
                    print(f"   프레임 수신 중... ({frame_count}개)")
        
        print(f"✅ 프레임 수신 테스트 완료! (총 {frame_count}개 프레임)")
        
        # 간단한 이미지 표시 테스트
        print("\n🖼️  이미지 표시 테스트 (ESC로 종료)...")
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if color_frame and depth_frame:
                # 이미지 변환
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Depth 이미지 컬러맵 변환
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), 
                    cv2.COLORMAP_JET
                )
                
                # 화면에 표시
                cv2.imshow('RealSense Color', color_image)
                cv2.imshow('RealSense Depth', depth_colormap)
                
                # ESC 키로 종료
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        
        return True
        
    except Exception as e:
        print(f"❌ RealSense 스트림 테스트 실패: {e}")
        print("\n가능한 원인:")
        print("1. 카메라가 다른 프로그램에서 사용 중")
        print("2. USB 대역폭 부족")
        print("3. 드라이버 문제")
        print("4. 권한 문제")
        return False
        
    finally:
        # 정리
        try:
            pipeline.stop()
            print("✅ 파이프라인 정상 종료")
        except:
            print("⚠️  파이프라인 종료 중 오류 (정상적일 수 있음)")
        
        cv2.destroyAllWindows()

def main():
    """메인 함수"""
    print("=" * 60)
    print("RealSense 카메라 단독 테스트 프로그램")
    print("=" * 60)
    
    # 1단계: 카메라 연결 확인
    if not test_realsense_connection():
        return
    
    # 2단계: 스트림 테스트
    if test_realsense_streams():
        print("\n🎉 RealSense 카메라 테스트 완료!")
        print("✅ 카메라가 정상적으로 작동합니다.")
    else:
        print("\n❌ RealSense 카메라 테스트 실패!")
        print("카메라 연결이나 설정을 확인해주세요.")

if __name__ == "__main__":
    main()
