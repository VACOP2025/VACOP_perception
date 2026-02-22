from launch import LaunchDescription
from launch_ros.actions import Node

# Topic du lidar
POINTCLOUD_TOPIC_NAME = '/rslidar_points'

def generate_launch_description():

    base_to_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_lidar',
        output='screen',
        arguments=['0.73', '0', '1.2', '0', '0.0174533', '0.0523599', 'base_link', 'rslidar']
    )

    odom_to_base = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='odom_to_base',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_link']
    )


    # Node RANSAC ground filter
    ground_filter = Node(
        package='lidar_ground_filter',
        executable='ransac_ground_filter_node',
        name='ransac_ground_filter',
        output='screen',
        parameters=[{
            'base_frame': 'base_link',
            'voxel_size': 0.02,
            'distance_threshold': 0.1,
            'plane_slope_threshold': 15.0,
        }],
        remappings=[
            ('input/points', POINTCLOUD_TOPIC_NAME)
        ]
    )

    # Node Euclidean clustering
    clustering = Node(
        package='lidar_ground_filter',
        executable='euclidean_cluster_node',
        name='euclidean_cluster',
        output='screen',
        parameters=[{
            'cluster_tolerance': 0.5,
            'min_cluster_size': 10,
        }],
        remappings=[
            ('input/no_ground', 'perception/scan/no_ground')
        ]
    )

    return LaunchDescription([
        odom_to_base,
        base_to_lidar,
        ground_filter,
        clustering
    ])
