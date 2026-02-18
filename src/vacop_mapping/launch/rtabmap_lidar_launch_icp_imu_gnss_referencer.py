import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    
    # --- CONFIGURATION ---
    lidar_topic = '/perception/scan/no_ground'
    imu_topic   = '/imu/data'
    gps_topic   = '/vacop/gnss/fix' #
    
    # --- FRAMES ---
    robot_frame = 'base_link' 
    odom_frame  = 'odom'
    
    # --- PARTIE 1 : Driver Lidar ---
    try:
        rslidar_dir = get_package_share_directory('rslidar_sdk')
        lidar_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(rslidar_dir, 'launch', 'start.py') 
            )
        )
    except Exception:
        lidar_launch = Node(package='dummy', executable='dummy') 

    # --- PARTIE 2 : TF STATIQUES ---
    lidar_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='lidar_tf_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'rslidar'] #
    )

    imu_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='imu_tf_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'imu_link'] #
    )

    gps_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='gps_tf_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'gnss'] #
    )

    # --- PARTIE 3 : KISS-ICP (Odométrie) ---
    # kiss_icp_node = Node(
    #     package='kiss_icp',
    #     executable='kiss_icp_node',
    #     name='kiss_icp_node',
    #     output='screen',
    #     parameters=[{
    #         'topic': lidar_topic,
    #         'visualize': False,
    #         'deskew': True,
    #         'base_frame': robot_frame,
    #         'odom_frame': odom_frame,
    #         'publish_odom_tf': True,
    #     }],
    #     remappings=[
    #         ('pointcloud_topic', lidar_topic),
    #         ('odometry', '/kiss/odometry'),
    #         ('imu', imu_topic)
    #     ]
    # )

    icp_odometry_node = Node(
        package='rtabmap_odom',
        executable='icp_odometry',
        output='screen',
        parameters=[{
            #'use_sim_time': use_sim_time,
            'frame_id': robot_frame,
            'odom_frame_id': 'odom',
            'publish_tf': True,
            'wait_for_transform': 0.2,
            'approx_sync': True,
            'queue_size': 20,
            
            # --- Réglages ROBUSTES pour Extérieur ---
            
            # --- Activation IMU pour Odométrie ---
            'wait_imu_to_init': True,
            'subscribe_imu': True,

            # Filtrage et Précision
            'Icp/VoxelSize': '0.35',          ### Augmenté (0.1 -> 0.2) pour réduire le bruit (herbe/feuilles) et le CPU
            'Icp/PointToPlane': 'true',
            'Icp/PointToPlaneK': '20',       ### AJOUTÉ : Nécessaire pour bien calculer les surfaces (murs/sol)
            
            # Tolérance de mouvement (Vitesse)
            'Icp/MaxCorrespondenceDistance': '1', ### AJOUTÉ : (0.1 -> 0.6) Crucial ! Tolère 60cm de décalage entre 2 scans
            'Icp/Iterations': '50',          ### Augmenté (10 -> 30) pour laisser le calcul converger
            'Icp/Epsilon': '0.001',          ### AJOUTÉ : Seuil de précision d'arrêt
            
            # Contraintes Physiques du Robot
            'Reg/Force3DoF': 'true',         ### AJOUTÉ : Bloque Z, Roll, Pitch. Le robot reste collé au sol.
            'Odom/GuessMotion': 'true',      ### AJOUTÉ : Utilise la vitesse précédente pour anticiper la suite
            'Odom/ResetCountdown': '1'       ### AJOUTÉ : Si perdu, tente de reset automatiquement
        }],
        remappings=[
            ('scan_cloud', lidar_topic), # CORRECTION: 'scan_cloud' au lieu de 'pointcloud_topic'
            ('odom', '/odom'),           # CORRECTION: Nom standard '/odom'
            ('imu', imu_topic)
        ]
    )

    # --- PARTIE 4 : RTAB-Map (SLAM Lidar avec Géoréférencement) ---
    slam_node = Node(
        package='rtabmap_slam', 
        executable='rtabmap', 
        output='screen',
        arguments=['--delete_db_on_start'], 
        parameters=[{
            'frame_id': robot_frame,      
            'map_frame_id': 'map',
            'odom_frame_id': odom_frame,  
            
            # --- Désactivation de la Vision ---
            'subscribe_depth': False,
            'subscribe_rgb': False,
            'subscribe_stereo': False,
            'subscribe_scan_cloud': True,
            
            # --- Odométrie & GPS ---
            'subscribe_odom': True,
            'odom_topic': '/odom', #
            'subscribe_gps': True, #
            
            # --- AJOUTS POUR LE GÉORÉFÉRENCEMENT ---
            'Rtabmap/LoopGPS': 'true', #
            'Rtabmap/GeoTargeting': 'true', 
            'Optimizer/PriorKnowledge': 'true',
            
            # --- IMU ---
            'wait_imu_to_init': True, #
            
            # --- Paramètres SLAM ---
            'Reg/Strategy': '1', # 1=ICP
            'Reg/Force3DoF': 'true', #
            'RGBD/ProximityBySpace': 'true', #
            'Grid/CellSize': '0.1', #
        }],
        remappings=[
            ('scan_cloud', lidar_topic), #
            ('gps/fix', gps_topic), #
            ('odom', '/odom'),
            ('imu', imu_topic) #
        ]
    )

    # # --- PARTIE 5 : Visualisation ---
    # viz_node = Node(
    #     package='rtabmap_viz',
    #     executable='rtabmap_viz',
    #     output='screen',
    #     parameters=[{
    #         'frame_id': robot_frame,
    #         'odom_frame_id': odom_frame,
    #         'subscribe_odom': True,
    #         'subscribe_scan_cloud': True,
    #         'subscribe_gps': True,
    #         'approx_sync': True
    #     }],
    #     remappings=[
    #         ('scan_cloud', lidar_topic),
    #         ('gps/fix', gps_topic),
    #         ('odom', '/kiss/odometry')
    #     ]
    # )

    return LaunchDescription([
        lidar_launch,
        lidar_tf,
        imu_tf,
        gps_tf,
        #kiss_icp_node,
        icp_odometry_node,
        slam_node,
        #viz_node
    ])
