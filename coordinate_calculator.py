#!/usr/bin/env python3

import numpy as np
import cv2

class CoordinateCalculator:
    def __init__(self, depth_scale=None):
        self.depth_scale = depth_scale
        self.use_undistortion = False
        self.use_color_matrix_for_3d = False
        print(f"🔧 좌표 계산기 초기화 - depth_scale: {depth_scale}")
        
    def get_3d_coordinates(self, bbox, depth_image, K_depth, dist_coeffs_depth, K_color, dist_coeffs_color):
        x1, y1, x2, y2 = bbox
        
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        
        if 0 <= center_x < depth_image.shape[1] and 0 <= center_y < depth_image.shape[0]:
            depth_raw = depth_image[center_y, center_x]
            
            if depth_raw > 0:
                depth_meters = depth_raw * self.depth_scale
                
                cx = K_depth[0, 2]
                cy = K_depth[1, 2]
                fx = K_depth[0, 0]
                fy = K_depth[1, 1]
                
                # 왜곡 보정 적용 여부
                if self.use_undistortion:
                    pixel_coords = np.array([[center_x, center_y]], dtype=np.float32)
                    undistorted_coords = cv2.undistortPoints(
                        pixel_coords.reshape(-1, 1, 2), 
                        K_color, 
                        dist_coeffs_color, 
                        P=K_color
                    ).reshape(-1, 2)
                    
                    final_x = undistorted_coords[0, 0]
                    final_y = undistorted_coords[0, 1]
                    pixel_info = f"픽셀: ({center_x}, {center_y}) → 왜곡보정: ({final_x:.1f}, {final_y:.1f})"
                else:
                    final_x = center_x
                    final_y = center_y
                    pixel_info = f"픽셀: ({center_x}, {center_y})"
                
                # 3D 계산용 매트릭스 선택
                if self.use_color_matrix_for_3d:
                    calc_cx = K_color[0, 2]
                    calc_cy = K_color[1, 2]
                    calc_fx = K_color[0, 0]
                    calc_fy = K_color[1, 1]
                    matrix_info = f"Color 매트릭스: fx={calc_fx:.1f}, fy={calc_fy:.1f}, cx={calc_cx:.1f}, cy={calc_cy:.1f}"
                else:
                    calc_cx = cx
                    calc_cy = cy
                    calc_fx = fx
                    calc_fy = fy
                    matrix_info = f"Depth 매트릭스: fx={calc_fx:.1f}, fy={calc_fy:.1f}, cx={calc_cx:.1f}, cy={calc_cy:.1f}"
                
                # 3D 좌표 계산
                x_3d = (final_x - calc_cx) * depth_meters / calc_fx
                y_3d = (final_y - calc_cy) * depth_meters / calc_fy
                z_3d = depth_meters
                
                print(f"📍 3D 좌표: ({x_3d*1000:.1f}, {y_3d*1000:.1f}, {z_3d*1000:.1f})mm")
                print(f"   📐 {pixel_info}")
                print(f"   🎯 {matrix_info}")
                
                x_3d_mm = x_3d * 1000
                y_3d_mm = y_3d * 1000
                z_3d_mm = z_3d * 1000
                
                return x_3d_mm, y_3d_mm, z_3d_mm
        
        print(f"❌ 유효하지 않은 픽셀 좌표: ({center_x}, {center_y})")
        return None, None, None
    
    def get_camera_matrices(self, color_intrinsics, depth_intrinsics):
        if color_intrinsics is None or depth_intrinsics is None:
            return None, None, None, None
            
        K_color = np.array([
            [color_intrinsics.fx, 0, color_intrinsics.ppx],
            [0, color_intrinsics.fy, color_intrinsics.ppy],
            [0, 0, 1]
        ])
        
        K_depth = np.array([
            [depth_intrinsics.fx, 0, depth_intrinsics.ppx],
            [0, depth_intrinsics.fy, depth_intrinsics.ppy],
            [0, 0, 1]
        ])
        
        dist_coeffs_color = np.array(color_intrinsics.coeffs)
        dist_coeffs_depth = np.array(depth_intrinsics.coeffs)
        
        print(f"🔧 카메라 매트릭스 생성:")
        print(f"   Depth fx: {depth_intrinsics.fx:.1f}, fy: {depth_intrinsics.fy:.1f}")
        print(f"   Depth cx: {depth_intrinsics.ppx:.1f}, cy: {depth_intrinsics.ppy:.1f}")
        print(f"   Depth 왜곡계수: {dist_coeffs_depth}")
        print(f"   Color 왜곡계수: {dist_coeffs_color}")
        
        return K_color, K_depth, dist_coeffs_color, dist_coeffs_depth
    
    def undistort_image(self, image, K, dist_coeffs):
        return cv2.undistort(image, K, dist_coeffs)
