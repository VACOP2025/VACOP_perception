# vacop_vision

Package ROS 2 de perception visuelle embarquée pour le robot VACOP, conçu pour tourner sur une Jetson (GPU CUDA). Il fusionne trois modèles d'IA accélérés via TensorRT pour détecter des objets, segmenter la zone roulable et classifier les feux tricolores en temps réel.

---

## Architecture du pipeline

```
Caméra USB (V4L2, 1280×720 @ 30 FPS)
        │
        ▼
┌────────────────────────┐
│    YOLO26m (TRT)       │  Détection : Personnes, Voitures, Panneaux Stop, Feux
├────────────────────────┤
│    TwinLiteNet (TRT)   │  Segmentation : Zone roulable + Lignes de voie
├────────────────────────┤
│    Classifier (TRT)    │  Classification Feux Tricolores : Rouge / Vert / Jaune
└────────────────────────┘
        │
        ▼
   Topics ROS 2
```

---

## Nœud principal : `vision_node.py`

**Nom du nœud :** `jetson_perception_node`  
**Point d'entrée :** `vision_node = vacop_vision.vision.vision_node:main`

### Description générale

Ce nœud implémente un pipeline de perception visuelle temps réel entièrement exécuté sur GPU. Il capture les images d'une caméra USB, les traite en parallèle avec trois modèles d'IA accélérés via TensorRT, puis publie les résultats sur ROS 2 à ~30 FPS.

### Pipeline d'inférence (par frame)

Chaque frame suit les étapes suivantes, exécutées dans un seul stream CUDA pour éviter les synchronisations inutiles :

```
Frame caméra (1280×720, BGR)
        │
        ├──► [1] YOLO26m (320×320, FP16)
        │         └─► Boîtes brutes : Person, Car, Stop Sign, Traffic Light
        │
        ├──► [2] TwinLiteNet (640×360, FP16)
        │         ├─► Masque "Drivable Area" (zone roulable)
        │         └─► Masque "Lane Lines" (lignes de voie)
        │
        └──► [3] Classifier (batch de crops 32×32, FP16)
                  └─► Couleur de chaque feu détecté : Rouge / Vert / Jaune
```

#### Étape 1 — Détection YOLO

- Le frame brut est passé directement à YOLO (resize interne à 320×320).
- Seuil de confiance : `CONF_THRESH = 0.25`.
- Classes filtrées : `0` (Person), `2` (Car), `9` (Traffic Light), `11` (Stop Sign). Les autres classes COCO sont ignorées.

#### Étape 2 — Segmentation TwinLiteNet

- Le frame est converti en tenseur GPU (`HWC → CHW`, normalisé `[0,1]`), puis redimensionné en `(360, 640)` via interpolation bilinéaire.
- TwinLiteNet produit deux sorties :
  - `da_predict` : carte de probabilité de la **zone roulable** (channel 1)
  - `ll_predict` : carte de probabilité des **lignes de voie** (channel 1)
- Seuillage à `0.5` pour binariser les masques.
- Le masque `drivable_area` est publié en `mono8` (0/255) sur `/perception/drivable_area` pour être exploité par Nav2.

#### Étape 3 — Classification des feux tricolores

- Pour chaque boîte YOLO de classe `9` (Traffic Light) :
  1. Le crop correspondant est extrait **directement depuis le tenseur GPU** (pas de copie CPU).
  2. Il est redimensionné à `(32, 32)` et normalisé `[-1, 1]`.
- Tous les crops sont regroupés en un **batch unique** et envoyés au classifieur en une seule passe TensorRT.
- Sortie : softmax sur 4 classes → `{1: Green, 2: Red, 3: Yellow}`.
- Seuil de confiance classifieur : `CLS_CONF_THRESH = 0.85` (volontairement élevé pour éviter les faux positifs).

### Post-traitement et publication

Une fois les inférences terminées et le stream CUDA synchronisé, le nœud :

1. **Mesure la latence GPU** via des CUDA Events (`start_evt` / `end_evt`) et l'affiche en HUD sur l'image de debug.

2. **Construit les messages ROS 2 :**
   - Chaque détection devient un `Detection2D` avec :
     - `bbox` : centre (x, y) + dimensions en pixels
     - `results[0].hypothesis.class_id` : identifiant textuel (`"person"`, `"car"`, `"stop_sign"`, `"traffic_light_red"`, etc.)
     - `results[0].hypothesis.score` : score de confiance YOLO
   - L'ensemble est empaqueté dans un `Detection2DArray` et publié sur `/perception/detections`.

3. **Construit l'image de debug** (`/perception/debug_view`) :
   - Superposition verte semi-transparente sur la zone roulable (alpha 0.4)
   - Superposition bleue semi-transparente sur les lignes de voie (alpha 0.5)
   - Boîtes de détection colorées par classe
   - Labels avec classe + score de confiance
   - HUD latence GPU en millisecondes

### Gestion de la caméra

La caméra est lue dans un **thread dédié** (`USBCameraReader`) découplé de la boucle d'inférence :
- Backend V4L2, format MJPEG, buffer réduit à 1 frame pour minimiser la latence.
- File d'attente de taille 2 : si l'inférence est plus lente que la caméra, l'ancien frame est éjecté — le nœud traite toujours le frame **le plus récent**.

### Initialisation GPU (Warmup)

Au démarrage, 5 passes à vide sont effectuées sur YOLO pour forcer la compilation JIT des kernels CUDA. Sans cette étape, la première frame réelle subirait une latence anormalement élevée (~500 ms).

### Topics publiés

| Topic | Type | Encoding | Description |
|---|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | `bgr8` | Image brute (désactivée par défaut) |
| `/perception/debug_view` | `sensor_msgs/Image` | `bgr8` | Image annotée complète |
| `/perception/drivable_area` | `sensor_msgs/Image` | `mono8` | Masque binaire zone roulable |
| `/perception/detections` | `vision_msgs/Detection2DArray` | — | Détections objets |

### Classes détectées

| `class_id` ROS | Classe réelle | Modèle source |
|---|---|---|
| `person` | Piéton | YOLO |
| `car` | Véhicule | YOLO |
| `stop_sign` | Panneau Stop | YOLO |
| `traffic_light_red` | Feu Rouge | YOLO + Classifier |
| `traffic_light_green` | Feu Vert | YOLO + Classifier |
| `traffic_light_yellow` | Feu Jaune | YOLO + Classifier |
| `traffic_light_unknown` | Feu (couleur indéterminée) | YOLO seul |

---

## Autres fichiers

### `vision_node_mqtt.py`

Variante de `vision_node.py` avec intégration d'un **dashboard distant**. En plus du pipeline de perception ROS 2, ce nœud se connecte à un broker MQTT (`neocampus.univ-tlse3.fr`) pour découvrir dynamiquement l'IP du dashboard, puis ouvre un flux vidéo GStreamer (H.264 over UDP, port 5001) vers celui-ci. À utiliser à la place de `vision_node.py` quand un monitoring distant est nécessaire.

### `TwinLite.py`

Définition PyTorch de l'architecture **TwinLiteNet** : réseau de segmentation légère multi-têtes comprenant des modules d'attention spatiale (`PAM_Module`) et de canal (`CAM_Module`), des blocs convolutifs (`CBR`, `CB`) et des couches de déconvolution (`UPx2`). Ce fichier est utilisé par `export_models.py` pour l'export vers ONNX/TensorRT.

### `export_models.py`

Script utilitaire (exécuté hors ROS) pour **convertir les modèles PyTorch en moteurs TensorRT** (`.engine`). Il exporte YOLO en FP16 directement via `ultralytics`, et fournit (en commentaire) les instructions pour exporter TwinLiteNet et le classifieur vers ONNX puis vers TensorRT via `trtexec`.

### `view_cam.py`

Script de test standalone pour **valider la caméra USB** : ouvre le flux V4L2 en MJPEG (1280×720, jusqu'à 60 FPS), affiche le flux dans une fenêtre avec le FPS mesuré en temps réel. Utile pour vérifier le câblage et les capacités de la caméra avant de lancer le nœud.

### `audit_thresholds.py`

Script d'évaluation offline pour **calibrer le seuil de confiance YOLO** sur le dataset de validation. Il fait varier le seuil de 0.05 à 0.60 et calcule Rappel, Précision et F1-Score pour les classes `Person` et `Car`, puis recommande le seuil optimal. À exécuter indépendamment du système ROS.

---

## Paramètres configurables

Tous les paramètres se trouvent en tête de `vacop_vision/vision/vision_node.py` sous la section `# --- CONFIGURATION CONSTANTES ---`. Aucune recompilation n'est nécessaire, il suffit d'éditer le fichier avant de lancer le nœud.

### Caméra

| Constante | Valeur défaut | Description |
|---|---|---|
| `CAMERA_INDEX` | `0` | Index du périphérique (`/dev/video0`). Changer si plusieurs caméras sont branchées. |
| `CAMERA_WIDTH` | `1280` | Largeur de capture en pixels. |
| `CAMERA_HEIGHT` | `720` | Hauteur de capture en pixels. |
| `CAMERA_FPS` | `30` | FPS demandé au driver V4L2. La valeur réelle dépend des capacités de la caméra. |

> Pour connaître les résolutions et FPS supportés par votre caméra :
> ```bash
> v4l2-ctl --device=/dev/video0 --list-formats-ext
> ```

### Modèles

| Constante | Valeur défaut | Description |
|---|---|---|
| `YOLO_ENGINE` | `models/yolo26m.engine` | Chemin vers le moteur TensorRT YOLO. Remplacer par un autre `.engine` si vous entraînez un modèle custom. |
| `TWINLITE_ENGINE` | `models/twinlite.engine` | Moteur TensorRT TwinLiteNet. |
| `CLASSIFIER_ENGINE` | `models/classifier.engine` | Moteur TensorRT du classifieur de feux. |
| `YOLO_IMG_SIZE` | `320` | Taille d'entrée YOLO en pixels (carré). Valeurs typiques : `320`, `416`, `640`. |

> Les chemins sont résolus **relativement au dossier du package** via `os.path`. Placez vos moteurs dans `vacop_vision/models/`.

### Seuils de confiance

| Constante | Valeur défaut | Effet si on augmente | Effet si on diminue |
|---|---|---|---|
| `CONF_THRESH` | `0.25` | Moins de détections, moins de faux positifs | Plus de détections, plus de faux positifs |
| `CLS_CONF_THRESH` | `0.85` | Classifieur de feux plus sélectif | Classifieur plus permissif (risque d'erreurs de couleur) |





---

## Lancement dans le conteneur Docker (seul, pour test)

```bash
ros2 run vacop_vision vision_node
```

---

