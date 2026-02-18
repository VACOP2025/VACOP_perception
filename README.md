# VACOP System

Système de perception et de cartographie autonome pour le robot **VACOP**, déployé sur une **Jetson Orin NX**. L'ensemble du système est containerisé via Docker et orchestré par Docker Compose.

---

## Architecture générale

Le système est composé de **deux modules indépendants** qui communiquent via ROS 2 (réseau hôte partagé) :

```
┌─────────────────────────────────────────────────────────┐
│                    Jetson Orin NX                        │
│                                                         │
│  ┌─────────────────────┐   ┌─────────────────────────┐  │
│  │  Container VISION   │   │  Container MAPPING      │  │
│  │  (vacop_vision)     │   │  (vacop_mapping)        │  │
│  │                     │   │                         │  │
│  │  Image: vacop:v3    │   │  Image: vacop_mapping:  │  │
│  │  Base: dusty-nv     │   │  latest                 │  │
│  │  (JetPack + CUDA +  │   │  Base: introlab3it/     │  │
│  │   TensorRT + ROS 2) │   │  rtabmap_ros:humble     │  │
│  │                     │   │                         │  │
│  │  • vision_node.py   │   │  • rslidar_sdk          │  │
│  │  • YOLO26m (TRT)    │   │  • vacop_mapping        │  │
│  │  • TwinLiteNet(TRT) │   │  • RTAB-Map             │  │
│  │  • Classifier (TRT) │   │  • Odométrie ICP        │  │
│  └─────────┬───────────┘   └───────────┬─────────────┘  │
│            │                           │                 │
│            └──────────┬────────────────┘                 │
│                       │  ROS 2 (network_mode: host)      │
│                  ROS_DOMAIN_ID=4                         │
└─────────────────────────────────────────────────────────┘
```

---

## Images Docker

### `vacop:v3` — Module Vision

| Propriété | Valeur |
|---|---|
| Image de base | [`dustynv/ros`](https://github.com/dusty-nv/jetson-containers) (JetPack) |
| Accélération | CUDA + TensorRT + cuDNN (natif JetPack) |
| ROS | ROS 2 Humble |
| Dockerfile | `Dockerfile.vision` |

`vacop:v3` est une image custom construite en amont  à partir de [`dustynv/ros`](https://github.com/dusty-nv/jetson-containers), l'image officielle NVIDIA pour Jetson. Elle embarque l'ensemble de la stack nécessaire au pipeline d'inférence IA :

- **CUDA / cuDNN** — accès direct au GPU intégré de la Jetson Orin NX (1024 cœurs CUDA, architecture Ampere)
- **TensorRT** — moteur d'inférence optimisé FP16, utilisé pour YOLO, TwinLiteNet et le classifieur de feux
- **PyTorch** — compilé pour ARM64 avec support CUDA, utilisé pour les pré/post-traitements GPU
- **Ultralytics (YOLO)** — chargement et warmup du modèle de détection
- **OpenCV** — capture V4L2, dessin des annotations et conversion d'images
- **ROS 2 Humble** — packages suppolémentaires, communication inter-modules

`Dockerfile.vision` (dans ce dépôt) se contente d'ajouter deux dépendances supplémentaires **par-dessus `vacop:v3`** :
- `python3-rclpy`
- `paho-mqtt` (pour la variante dashboard `vision_node_mqtt.py`)

### `vacop_mapping:latest` — Module Mapping

| Propriété | Valeur |
|---|---|
| Image de base | [`introlab3it/rtabmap_ros:humble-latest`](https://github.com/introlab/rtabmap_ros) |
| ROS | ROS 2 Humble |
| Dockerfile | `Dockerfile.mapping` |

`Dockerfile.mapping` ajoute :
- `libpcap-dev` (driver LiDAR Robosense)
- `python3-paho-mqtt`
- `ros-humble-navigation2` / `nav2-bringup`
- `ros-humble-ros2-control` / `ros2-controllers`

---

## Packages ROS 2

| Package | Container | Description |
|---|---|---|
| `vacop_vision` | vision | Pipeline de perception caméra (YOLO + TwinLiteNet + Classifier) |
| `vacop_mapping` | mapping | Cartographie et localisation via RTAB-Map + LiDAR Robosense |
| `lidar_ground_filter` | mapping | Filtrage du sol LiDAR (RANSAC) et clustering d'obstacles |

---

## Lancement

### Prérequis

- Jetson Orin NX avec JetPack installé
- Docker et NVIDIA Container Runtime configurés
- Workspace ROS 2 cloné dans `~/vacop_jetson`

```bash
# Vérifier que le runtime NVIDIA est disponible
docker info | grep -i runtime
```

### Démarrer les deux modules

```bash
cd /path/to/vacop_system
docker compose up
```

Pour lancer en arrière-plan :

```bash
docker compose up -d
```

### Lancer un seul module

```bash
# Vision uniquement
docker compose up vision

# Mapping uniquement
docker compose up mapping
```

### Arrêter

```bash
docker compose down
```

---

## Configuration Docker Compose

Les deux services partagent les réglages suivants (voir `compose.yaml`) :

| Paramètre | Valeur | Raison |
|---|---|---|
| `network_mode` | `host` | Les topics ROS 2 sont partagés sans bridge réseau |
| `ipc` | `host` | Mémoire partagée pour la communication inter-process ROS 2 |
| `runtime` | `nvidia` | Accès GPU CUDA depuis le conteneur |
| `ROS_DOMAIN_ID` | `4` | Isolation du réseau ROS 2 sur le LAN |
| `privileged` | `true` | Accès aux périphériques matériels (caméra, LiDAR) |

### Volume principal

```yaml
~/vacop_jetson:/root/vacop_ws
```

Le workspace ROS 2 est monté depuis l'hôte dans les deux conteneurs. Les packages sont donc partagés et modifiables sans reconstruire l'image.

---

## Topics ROS 2 inter-modules

Les deux conteneurs communiquent via les topics suivants (réseau hôte partagé) :

| Topic | Producteur | Consommateur | Type |
|---|---|---|---|
| `/perception/drivable_area` | vision | mapping / Nav2 | `sensor_msgs/Image` |
| `/perception/detections` | vision | mapping | `vision_msgs/Detection2DArray` |
| `/perception/debug_view` | vision | — (debug) | `sensor_msgs/Image` |
| `/rslidar_points` | mapping (driver) | mapping (RTAB-Map) | `sensor_msgs/PointCloud2` |
| `/map` | mapping | — | `nav_msgs/OccupancyGrid` |
| `/tf` / `/tf_static` | mapping | vision (optionnel) | `tf2_msgs/TFMessage` |

---

## Structure du dépôt

```
vacop_system/
├── compose.yaml              # Orchestration Docker Compose
├── Dockerfile.vision         # Image vision (basée sur dustynv/ros)
├── Dockerfile.mapping        # Image mapping (basée sur introlab/rtabmap_ros)
└── src/
    ├── vacop_vision/         # Package ROS 2 pour le module de vision
    ├── vacop_mapping/        # Package ROS 2 pour la cartographie
    └── lidar_ground_filter/  # Package ROS 2 pour le filtrage sol LiDAR
```

Chaque package possède son propre `README.md` détaillant son fonctionnement interne.

---

## Matériel cible

| Composant | Modèle |
|---|---|
| Calculateur | NVIDIA Jetson Orin NX |
| Caméra | Caméra USB V4L2 (1280×720 @ 30 FPS) |
| LiDAR | Robosense RS-LiDAR |
| IMU | Intégrée / externe sur `imu_link` |

---




