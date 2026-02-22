import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

# Prototype non testé !!!

def generate_launch_description():
    extrinsics_path = os.path.expanduser('parameters/extrinsics_lidar_to_camera_pair_0004.json')
    intrinsics_path = os.path.expanduser('parameters/calibration.ymgit l')
    
    return LaunchDescription([
        # Nœud de Perception Caméra 
        Node(
            package='vacop_vision',
            executable='vision_node.py',
            name='vision_node',
            output='screen',
            parameters=[{
                'use_sim_time': False
            }]
        ),
        
        # Nœud de Fusion LiDAR-Caméra
        Node(
            package='vacop_fusion',
            executable='lidar_camera_fusion_node.py',
            name='lidar_camera_fusion',
            output='screen',
            parameters=[{
                'extrinsics_path': extrinsics_path,
                'intrinsics_path': intrinsics_path,
                'use_best_pair': True,
                'publish_fused_cloud': True,
                'publish_obstacle_map': True,
                'detection_depth_threshold': 15.0,     # 15m max
                'obstacle_inflation_radius': 0.8,      # 80cm d'inflation
                # Paramètres Drivable Area
                'use_drivable_area': True,             # Activer la zone roulable
                'drivable_area_cost_weight': 50,       # Coût zones non-roulables
                'drivable_projection_distance': 10.0,  # Distance projection (m)
            }]
        ),
    ])
