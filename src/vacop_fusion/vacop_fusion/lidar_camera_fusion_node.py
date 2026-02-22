#!/usr/bin/env python3
"""
Prototype python pour un nœud ROS2 de Fusion LiDAR-Caméra : non testé !! 
Projette les détections caméra dans l'espace LiDAR et enrichit Nav2
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from nav_msgs.msg import OccupancyGrid
import sensor_msgs_py.point_cloud2 as pc2
from cv_bridge import CvBridge

import numpy as np
import yaml
import json
from collections import deque
from threading import Lock


class LiDARCameraFusion(Node):
    def __init__(self):
        super().__init__('lidar_camera_fusion')
        
        self.get_logger().info("Initialisation du nœud de Fusion LiDAR-Caméra...")
        
        # Paramètres configurables
        self.declare_parameter('extrinsics_path', 'parameters/extrinsics_lidar_to_camera_pair_0004.json')
        self.declare_parameter('intrinsics_path', 'parameters/calibration.yml')
        self.declare_parameter('use_best_pair', True)
        self.declare_parameter('publish_fused_cloud', True)
        self.declare_parameter('publish_obstacle_map', True)
        self.declare_parameter('use_drivable_area', True)          # la zone roulable
        self.declare_parameter('drivable_area_cost_weight', 50)    # Coût pour zones non-roulables
        self.declare_parameter('detection_depth_threshold', 15.0)  # mètres
        self.declare_parameter('obstacle_inflation_radius', 0.5)   # mètres
        self.declare_parameter('drivable_projection_distance', 10.0) # Distance de projection (m)
        
        # Chargement des calibrations
        self.load_calibrations()
        
        # Buffer pour synchronisation temporelle (approximative)
        self.detections_buffer = deque(maxlen=5)
        self.lidar_buffer = deque(maxlen=3)
        self.drivable_buffer = deque(maxlen=3)  # Buffer pour masque de route
        self.latest_drivable_mask = None        # Dernier masque valide
        self.lock = Lock()
        
        self.bridge = CvBridge()
        
        # Subscribers
        self.sub_detections = self.create_subscription(
            Detection2DArray, '/perception/detections', 
            self.detections_callback, 10)
        
        self.sub_lidar = self.create_subscription(
            PointCloud2, 'rslidar_points', 
            self.lidar_callback, 10)
        
        self.sub_drivable = self.create_subscription(
            Image, '/perception/drivable_area',
            self.drivable_callback, 10)
        
        # Publishers
        if self.get_parameter('publish_fused_cloud').value:
            self.pub_fused_cloud = self.create_publisher(
                PointCloud2, '/fusion/annotated_cloud', 10)
        
        if self.get_parameter('publish_obstacle_map').value:
            self.pub_obstacle_map = self.create_publisher(
                OccupancyGrid, '/fusion/obstacle_map', 10)
        
        # Pour Nav2 : publier des obstacles dynamiques
        self.pub_obstacle_layer = self.create_publisher(
            OccupancyGrid, '/local_costmap/obstacle_layer', 10)
        
        # Timer pour fusion périodique
        self.timer = self.create_timer(0.1, self.fusion_callback)  # 10 Hz
        
        self.get_logger().info("Fusion LiDAR-Caméra prête ✅")
    
    def load_calibrations(self):
        """Charge les paramètres de calibration extrinsèques et intrinsèques"""
        # Extrinsèques (LiDAR -> Caméra)
        extr_path = self.get_parameter('extrinsics_path').value
        extr_path = extr_path.replace('~', '/vacop_ws') 
        
        try:
            with open(extr_path, 'r') as f:
                extr_data = json.load(f)
            
            self.T_lidar_to_cam = np.array(extr_data['T_lidar_to_camera'], dtype=np.float64)
            self.R_lidar_to_cam = self.T_lidar_to_cam[:3, :3]
            self.t_lidar_to_cam = self.T_lidar_to_cam[:3, 3]
            
            self.get_logger().info(f"Extrinsèques chargés depuis {extr_path}")
            self.get_logger().info(f"Translation LiDAR->Cam: {self.t_lidar_to_cam}")
            
        except Exception as e:
            self.get_logger().error(f"Erreur chargement extrinsèques: {e}")
            raise
        
        # Intrinsèques caméra
        intr_path = self.get_parameter('intrinsics_path').value
        intr_path = intr_path.replace('~', '/home/jetson')
        
        try:
            with open(intr_path, 'r') as f:
                intr_yaml = yaml.safe_load(f)
            
            self.K = np.array(intr_yaml['K']['data'], dtype=np.float64).reshape(3, 3)
            self.D = np.array(intr_yaml['dist']['data'], dtype=np.float64)
            self.img_width = intr_yaml['image_width']
            self.img_height = intr_yaml['image_height']
            
            self.get_logger().info(f"Intrinsèques chargés: {self.img_width}x{self.img_height}")
            
        except Exception as e:
            self.get_logger().error(f"Erreur chargement intrinsèques: {e}")
            raise
    
    def detections_callback(self, msg):
        """Réception des détections visuelles"""
        with self.lock:
            self.detections_buffer.append(msg)
    
    def lidar_callback(self, msg):
        """Réception du nuage de points LiDAR"""
        with self.lock:
            self.lidar_buffer.append(msg)
    
    def drivable_callback(self, msg):
        """Masque de zone roulable (TwinLiteNet)"""
        try:
            # Conversion du message ROS en numpy array
            drivable_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            
            with self.lock:
                self.drivable_buffer.append(drivable_mask)
                self.latest_drivable_mask = drivable_mask
                
        except Exception as e:
            self.get_logger().error(f"Erreur traitement drivable area: {e}")
    
    def project_lidar_to_image(self, points_lidar):
        """
        Projette des points 3D LiDAR dans l'image caméra
        
        Args:
            points_lidar: (N, 3) array de points en repère LiDAR
        
        Returns:
            pixels: (N, 2) coordonnées dans l'image
            depths: (N,) profondeur en repère caméra
            valid_mask: (N,) booléen des points visibles
        """
        # Transformation LiDAR -> Caméra
        points_cam = (self.R_lidar_to_cam @ points_lidar.T).T + self.t_lidar_to_cam[None, :]
        
        # Filtrer les points derrière la caméra
        valid_depth = points_cam[:, 2] > 0.1
        
        # Projection perspective (sans distorsion pour l'instant)
        pixels = np.zeros((len(points_cam), 2))
        pixels[:, 0] = (self.K[0, 0] * points_cam[:, 0] / points_cam[:, 2]) + self.K[0, 2]
        pixels[:, 1] = (self.K[1, 1] * points_cam[:, 1] / points_cam[:, 2]) + self.K[1, 2]
        
        # Filtrer les points hors image
        valid_x = (pixels[:, 0] >= 0) & (pixels[:, 0] < self.img_width)
        valid_y = (pixels[:, 1] >= 0) & (pixels[:, 1] < self.img_height)
        valid_mask = valid_depth & valid_x & valid_y
        
        return pixels, points_cam[:, 2], valid_mask
    
    def project_drivable_area_to_ground(self, drivable_mask):
        """
        Projette le masque de zone roulable (2D image) sur un grid 3D au sol
        
        Args:
            drivable_mask: (H, W) array binaire (255 = roulable, 0 = non-roulable)
        
        Returns:
            ground_grid: (grid_h, grid_w) array avec probabilité de zone roulable
        """
        if drivable_mask is None:
            return None
        
        # Paramètres du grid au sol (dans le repère base_link)
        grid_resolution = 0.1  # 10cm
        grid_width = 100       # 10m de large (5m de chaque côté)
        grid_height = 200      # 20m devant le robot
        max_distance = self.get_parameter('drivable_projection_distance').value
        
        # Origine du grid
        origin_x = -grid_width * grid_resolution / 2
        origin_y = 0.0
        
        # Grille de votes (accumulation)
        ground_grid = np.zeros((grid_height, grid_width), dtype=np.float32)
        vote_count = np.zeros((grid_height, grid_width), dtype=np.int32)
        
        # Échantillonnage du grid au sol
        # Pour chaque cellule, on projette son centre dans l'image
        for i in range(0, grid_height, 2):  # Sous-échantillonnage pour perfs
            for j in range(0, grid_width, 2):
                # Position 3D au sol dans le repère base_link
                x = origin_x + (j + 0.5) * grid_resolution
                y = origin_y + (i + 0.5) * grid_resolution
                z = 0.0  # Au sol
                
                # Distance au robot
                dist = np.sqrt(x**2 + y**2)
                if dist > max_distance:
                    continue
                
                # Point 3D en repère LiDAR (approximation: base_link ≈ lidar_link)
                # TODO: lidar_link != base_link, il faut lire la tf publiée 
                pt_lidar = np.array([x, y, z])
                
                # Projection dans l'image
                pixels, depths, valid = self.project_lidar_to_image(pt_lidar[None, :])
                
                if not valid[0]:
                    continue
                
                u, v = int(pixels[0, 0]), int(pixels[0, 1])
                
                # Vérifier si ce point est dans une zone roulable
                if 0 <= v < drivable_mask.shape[0] and 0 <= u < drivable_mask.shape[1]:
                    is_drivable = drivable_mask[v, u] > 127  # Seuil
                    
                    # Remplir la cellule et ses voisines (pour combler les trous)
                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < grid_height and 0 <= nj < grid_width:
                                ground_grid[ni, nj] += 1.0 if is_drivable else 0.0
                                vote_count[ni, nj] += 1
        
        # Normalisation par le nombre de votes
        mask = vote_count > 0
        ground_grid[mask] /= vote_count[mask]
        
        return ground_grid
    
    def associate_detections_with_lidar(self, detections, lidar_msg):
        """
        Associe les détections 2D avec les points LiDAR
        
        Returns:
            obstacles_3d: Liste de (class_id, x, y, z, confidence)
        """
        # Conversion du nuage de points en numpy array
        points_lidar = []
        for point in pc2.read_points(lidar_msg, field_names=("x", "y", "z"), skip_nans=True):
            points_lidar.append([point[0], point[1], point[2]])
        
        if len(points_lidar) == 0:
            return []
        
        points_lidar = np.array(points_lidar, dtype=np.float64)
        
        # Projection dans l'image
        pixels, depths, valid_mask = self.project_lidar_to_image(points_lidar)
        
        # Pour chaque détection, trouver les points LiDAR associés
        obstacles_3d = []
        
        for det in detections.detections:
            if len(det.results) == 0:
                continue
            
            # Récupération des paramètres de la détection
            class_id = det.results[0].hypothesis.class_id
            confidence = det.results[0].hypothesis.score
            
            # Bounding box (centre et taille)
            cx = det.bbox.center.x
            cy = det.bbox.center.y
            w = det.bbox.size_x
            h = det.bbox.size_y
            
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            
            # Trouver les points LiDAR dans cette bbox
            in_box = (pixels[:, 0] >= x1) & (pixels[:, 0] <= x2) & \
                     (pixels[:, 1] >= y1) & (pixels[:, 1] <= y2) & valid_mask
            
            if not np.any(in_box):
                continue
            
            # Points 3D associés (en repère LiDAR)
            associated_points = points_lidar[in_box]
            
            # Position 3D médiane de l'obstacle
            pos_3d = np.median(associated_points, axis=0)
            
            # Filtre de distance maximale
            dist = np.linalg.norm(pos_3d[:2])  # Distance planaire
            if dist > self.get_parameter('detection_depth_threshold').value:
                continue
            
            obstacles_3d.append({
                'class_id': class_id,
                'position': pos_3d,
                'confidence': confidence,
                'points': associated_points
            })
        
        return obstacles_3d
    
    def create_obstacle_map(self, obstacles_3d, drivable_ground_grid=None, reference_frame='base_link'):
        """
        Crée une OccupancyGrid pour Nav2 avec les obstacles détectés ET la zone roulable
        
        Args:
            obstacles_3d: Liste d'obstacles 3D
            drivable_ground_grid: Grid de probabilité de zone roulable (optionnel)
            reference_frame: Frame de référence (base_link ou map)
        
        Returns:
            OccupancyGrid message
        """
        # Paramètres de la grille locale (à ajuster)
        grid_resolution = 0.1  # 10 cm par cellule
        grid_width = 100       # 10m de large
        grid_height = 200      # 20m de long (devant le robot)
        
        # Origine de la grille (centrée sur le robot)
        origin_x = -grid_width * grid_resolution / 2
        origin_y = 0.0  # Commence au robot
        
        # Création de la grille (0 = libre, 100 = occupé)
        grid = np.zeros((grid_height, grid_width), dtype=np.int8)
        
        # ÉTAPE 1: Appliquer la drivable area si disponible
        if drivable_ground_grid is not None and self.get_parameter('use_drivable_area').value:
            cost_weight = self.get_parameter('drivable_area_cost_weight').value
            
            # Convertir les probabilités en coûts
            # prob = 1.0 → coût = 0 (libre)
            # prob = 0.0 → coût = cost_weight (non-roulable)
            non_drivable_cost = (1.0 - drivable_ground_grid) * cost_weight
            grid = np.clip(non_drivable_cost, 0, 100).astype(np.int8)
        
        # ÉTAPE 2: Ajouter les obstacles détectés par la caméra
        inflation_radius = self.get_parameter('obstacle_inflation_radius').value
        inflation_cells = int(inflation_radius / grid_resolution)
        
        for obs in obstacles_3d:
            x, y, z = obs['position']
            
            # Conversion en indices de grille
            i = int((y - origin_y) / grid_resolution)
            j = int((x - origin_x) / grid_resolution)
            
            if 0 <= i < grid_height and 0 <= j < grid_width:
                # Marquer comme occupé avec inflation
                for di in range(-inflation_cells, inflation_cells + 1):
                    for dj in range(-inflation_cells, inflation_cells + 1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < grid_height and 0 <= nj < grid_width:
                            # Distance à l'obstacle
                            dist = np.sqrt(di**2 + dj**2) * grid_resolution
                            if dist <= inflation_radius:
                                # Coût décroissant avec la distance
                                cost = int(100 * (1 - dist / inflation_radius))
                                # Prendre le max entre obstacle et non-drivable
                                grid[ni, nj] = max(grid[ni, nj], cost)
        
        # Création du message ROS
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = reference_frame
        
        msg.info.resolution = grid_resolution
        msg.info.width = grid_width
        msg.info.height = grid_height
        msg.info.origin.position.x = origin_x
        msg.info.origin.position.y = origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        
        msg.data = grid.flatten().tolist()
        
        return msg
    
    def fusion_callback(self):
        """Callback principal de fusion (10 Hz)"""
        with self.lock:
            if len(self.detections_buffer) == 0 or len(self.lidar_buffer) == 0:
                return
            
            # Récupération des dernières données (synchronisation simple)
            latest_detections = self.detections_buffer[-1]
            latest_lidar = self.lidar_buffer[-1]
            latest_drivable = self.latest_drivable_mask  # Peut être None
        
        # Association détections <-> LiDAR
        obstacles_3d = self.associate_detections_with_lidar(latest_detections, latest_lidar)
        
        # Projection de la drivable area sur le sol
        drivable_ground_grid = None
        if latest_drivable is not None and self.get_parameter('use_drivable_area').value:
            drivable_ground_grid = self.project_drivable_area_to_ground(latest_drivable)
            if drivable_ground_grid is not None:
                self.get_logger().info("Drivable area projetée sur le sol", throttle_duration_sec=5.0)
        
        if len(obstacles_3d) > 0:
            self.get_logger().info(f"Fusion: {len(obstacles_3d)} obstacles 3D détectés")
        
        # Publication de la carte d'obstacles pour Nav2
        if self.get_parameter('publish_obstacle_map').value:
            obstacle_map = self.create_obstacle_map(obstacles_3d, drivable_ground_grid)
            self.pub_obstacle_layer.publish(obstacle_map)
        


def main(args=None):
    rclpy.init(args=args)
    node = LiDARCameraFusion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
