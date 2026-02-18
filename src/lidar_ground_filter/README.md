# Lidar Ground Filter

Ce package ROS 2 implémente une pipeline de perception LiDAR pour la détection d'obstacles et le filtrage du sol. Il se compose de deux nœuds principaux :
1.  **RANSAC Ground Filter** : Segmente le nuage de points pour séparer le sol des obstacles.
2.  **Euclidean Cluster** : Regroupe les points d'obstacles pour détecter des objets distincts.

## Architecture des Nœuds

### 1. `ransac_ground_filter_node`
Ce nœud filtre le sol en utilisant l'algorithme RANSAC (Random Sample Consensus).
* **Entrée** : Nuage de points brut.
* **Traitement** : Transforme le nuage dans le repère du robot, applique un filtre VoxelGrid, détecte le plan du sol et vérifie sa pente.
* **Sortie** : Deux nuages de points (`ground` et `no_ground`).

### 2. `euclidean_cluster_node`
Ce nœud détecte les objets à partir des points restants (obstacles).
* **Entrée** : Nuage de points sans le sol (`no_ground`).
* **Traitement** : Utilise une extraction euclidienne (clustering) via un KD-Tree pour grouper les points proches.
* **Sortie** : `MarkerArray` (bounding boxes) pour la visualisation.

---

## Configuration du Launch File (`ground_filter.launch.py`)

Le comportement du système est défini par les paramètres passés aux nœuds lors du lancement.

### Paramètres de Filtrage du Sol (`ransac_ground_filter_node`)

| Paramètre | Valeur par défaut | Description et Impact |
| :--- | :--- | :--- |
| `base_frame` | `base_link` | **Repère de référence**. Le nuage est transformé dans ce repère pour que le sol soit horizontal (Z=0). |
| `voxel_size` | `0.03` | **Taille des voxels (en m)**. Définit la résolution du sous-échantillonnage. <br>_Impact :_ Une valeur plus élevée accélère le calcul mais réduit la précision. |
| `distance_threshold` | `0.08` | **Seuil de distance (en m)**. Épaisseur tolérée pour considérer un point comme appartenant au sol. <br>_Impact :_ Trop bas (<0.05), le filtre rate des parties du sol. Trop haut (>0.15), il efface les trottoirs. |
| `plane_slope_threshold` | `15.0` | **Pente maximale (en degrés)**. Angle max autorisé entre le plan détecté et l'horizontale. <br>_Impact :_ Évite de détecter des murs ou des rampes raides comme étant du sol. |

### Paramètres de Clustering (`euclidean_cluster_node`)

| Paramètre | Valeur par défaut | Description et Impact |
| :--- | :--- | :--- |
| `cluster_tolerance` | `0.5` | **Tolérance de distance (en m)**. Distance maximale entre deux points pour appartenir au même objet. <br>_Impact :_ Augmenter cette valeur fusionne des objets proches. La diminuer sépare les objets mais peut fragmenter un objet unique. |
| `min_cluster_size` | `10` | **Taille minimale du cluster**. Nombre minimum de points pour valider un objet. <br>_Impact :_ Filtre le "bruit" (points isolés). |

### Transformations Statiques (TF)

Le fichier de lancement publie également des TFs statiques nécessaires au bon fonctionnement si le robot ne les fournit pas :
* `base_to_lidar` : Position du LiDAR par rapport à la base du robot (x=0.73m, z=1.2m).
* `odom_to_base` : Lien odométrie (utile pour la visualisation fixe).

---

## Installation et Utilisation

### Compilation

```
cd ~/ros2_ws
colcon build --packages-select lidar_ground_filter
source install/setup.bash
```

### Lancement

Utilisez le fichier launch qui contient les paramètres optimisés :

```
ros2 launch lidar_ground_filter ground_filter.launch.py
```

## Topics ROS 2

| Type | Topic | Description |
| :--- | :--- | :--- |
| **Sub** | `/rslidar_points` | Nuage de points brut du LiDAR (Entrée remappée). |
| **Pub** | `perception/scan/ground` | Points classifiés comme sol (sortie du filtre RANSAC). |
| **Pub** | `perception/scan/no_ground` | Points classifiés comme obstacles (sortie du filtre RANSAC, entrée du clustering). |
| **Pub** | `perception/scan/cluster3D` | Visualisation des objets détectés (Markers pour RViz). |


