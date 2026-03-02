import os
import sys
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    package_name = 'orbbec_camera'
    launch_file_name = 'gemini_330_series.launch.py'
    
    orbbec_launch_file = os.path.join(
        get_package_share_directory(package_name),
        'launch',
        launch_file_name
    )

    crop_script_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 
        'crop_node.py'
    )

    # ================= 설정 값 =================
    # 하드웨어: 640x400 @ 10fps -> Python: 480x426 변환
    req_width = '640'
    req_height = '400'
    req_fps = '10'
    
    # [최적화] RGB 외 불필요한 기능 All Off (대역폭 절약)
    common_params = {
        # --- [1] 기본 센서 끄기 ---
        'enable_point_cloud': 'false',
        'enable_colored_point_cloud': 'false',
        'enable_depth': 'false',
        'enable_infra1': 'false',
        'enable_infra2': 'false',
        'enable_accel': 'false',
        'enable_gyro': 'false',
        'enable_audio': 'false',
        
        # --- [2] 잡다한 토픽/기능 제거 (여기가 핵심) ---
        'publish_tf': 'false',              # TF 좌표계 제거
        'enable_publish_extrinsic': 'false',
        'enable_d2c_viewer': 'false',
        'enable_metadata': 'false',         # metadata 토픽 제거
        'enable_soft_filter': 'false',      # depth_filter_status 제거
        'diagnostic_publish_rate': '0.0',   # device_status 제거 (0.0=끄기)
        
        # --- [3] 해상도 설정 ---
        'color_width': req_width,
        'color_height': req_height,
        'color_fps': req_fps,
    }
    # ==========================================

    # ---------------------------------------------------------
    # 1. Front Camera (G5) - 즉시 실행
    # ---------------------------------------------------------
    ns_front = 'cam_front'
    
    # 파라미터 복사 후 시리얼/이름 설정
    front_params = common_params.copy()
    front_params.update({
        'camera_name': ns_front,
        'serial_number': 'CP82841000G5',  # [Front]
        'device_num': '1'
    })

    cam_front_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(orbbec_launch_file),
        launch_arguments=front_params.items()
    )
    
    cam_front_cropper = Node(
        package=None, executable=sys.executable,
        arguments=[crop_script_path],
        namespace=ns_front,
        output='screen',
        parameters=[{'target_w': 480, 'target_h': 426}]
    )

    # ---------------------------------------------------------
    # 2. Right Camera (C2) - 2초 딜레이
    # ---------------------------------------------------------
    ns_right = 'cam_right'

    right_params = common_params.copy()
    right_params.update({
        'camera_name': ns_right,
        'serial_number': 'CP82841000KH',  # [Right]CP82841000KH
        'device_num': '2'
    })

    cam_right_driver = TimerAction(
        period=4.0,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(orbbec_launch_file),
            launch_arguments=right_params.items()
        )]
    )

    cam_right_cropper = Node(
        package=None, executable=sys.executable,
        arguments=[crop_script_path],
        namespace=ns_right,
        output='screen',
        parameters=[{'target_w': 480, 'target_h': 426}]
    )

    # ---------------------------------------------------------
    # 3. Left Camera (KH) - 4초 딜레이
    # ---------------------------------------------------------
    ns_left = 'cam_left'

    left_params = common_params.copy()
    left_params.update({
        'camera_name': ns_left,
        'serial_number': 'CP82841000C2',  # [Left]
        'device_num': '3'
    })

    cam_left_driver = TimerAction(
        period=8.0,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(orbbec_launch_file),
            launch_arguments=left_params.items()
        )]
    )

    cam_left_cropper = Node(
        package=None, executable=sys.executable,
        arguments=[crop_script_path],
        namespace=ns_left,
        output='screen',
        parameters=[{'target_w': 480, 'target_h': 426}]
    )

    return LaunchDescription([
        cam_front_driver, cam_front_cropper,
        cam_right_driver, cam_right_cropper,
        cam_left_driver, cam_left_cropper
    ])