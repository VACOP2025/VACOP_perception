#!/usr/bin/env python3
"""
Nœud ROS 2 de Perception : fusion YOLO26 + TwinLiteNet + Classifier.
Publie : images, masques de route, et détections d'objets.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge

import cv2
import torch
import numpy as np
import time
import threading
import queue
import tensorrt as trt
from ultralytics import YOLO
import torchvision.transforms as T
import gc
import os
# --- CONFIGURATION CONSTANTES ---
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# Chemins des modèles
#WORKSPACE_PATH = "/root/vacop_ws/src/vacop_vision"
WORKSPACE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_ENGINE = os.path.join(WORKSPACE_PATH, "models/yolo26m.engine")
TWINLITE_ENGINE = os.path.join(WORKSPACE_PATH, "models/twinlite.engine")
CLASSIFIER_ENGINE = os.path.join(WORKSPACE_PATH, "models/classifier.engine")

CONF_THRESH = 0.25
CLS_CONF_THRESH = 0.85
YOLO_IMG_SIZE = 320

COLORS = {
    'Red': (0, 0, 255), 'Green': (0, 255, 0), 'Yellow': (0, 255, 255),
    'Car': (100, 100, 100), 'Person': (255, 100, 0), 'Stop': (0, 0, 128)
}

# --- CLASSES UTILITAIRES (Moteur TRT & Caméra) ---

class TRTModuleStatic:
    """Wrapper pour l'inférence TensorRT brute"""
    def __init__(self, engine_path, device):
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        try:
            with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        except FileNotFoundError:
            print(f"ERREUR CRITIQUE: Le moteur {engine_path} est introuvable !")
            raise

        self.context = self.engine.create_execution_context()
        self.inputs = []
        self.outputs = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            
            # Gestion des dimensions dynamiques (-1)
            if -1 in shape:
                 real_shape = list(shape)
                 for dim_idx, dim in enumerate(real_shape):
                     if dim == -1: real_shape[dim_idx] = 16 
                 tensor = torch.zeros(tuple(real_shape), device=device, dtype=torch.float32)
            else:
                 tensor = torch.zeros(tuple(shape), device=device, dtype=torch.float32)

            self.context.set_tensor_address(name, tensor.data_ptr())
            
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'name': name, 'tensor': tensor})
            else:
                self.outputs.append({'name': name, 'tensor': tensor})

    def forward(self, x, stream):
        input_info = self.inputs[0]
        # Redimensionnement si nécessaire (contexte dynamique)
        if x.shape != input_info['tensor'].shape:
            self.context.set_input_shape(input_info['name'], x.shape)
        
        self.context.set_tensor_address(input_info['name'], x.data_ptr())
        self.context.execute_async_v3(stream.cuda_stream)
        return [out['tensor'] for out in self.outputs]


class USBCameraReader:
    """Lecteur caméra threadé"""
    def __init__(self, camera_index=0, width=1280, height=720, fps=30, queue_size=2):
        self.stopped = False
        self.q = queue.Queue(maxsize=queue_size)
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        
        if not self.cap.isOpened():
            print(f"Attention: Impossible d'ouvrir la caméra {camera_index}")
            self.working = False
            return
        else:
            self.working = True

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.t = threading.Thread(target=self.update, daemon=True)
        self.t.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            if self.q.full():
                try: self.q.get_nowait()
                except queue.Empty: pass
            self.q.put(frame)
        self.cap.release()

    def read(self):
        try: return self.q.get(timeout=1.0)
        except queue.Empty: return None

    def stop(self):
        self.stopped = True
        if self.working:
            self.t.join(timeout=1.0)

# --- NOEUD ROS 2 ---

class JetsonPerceptionNode(Node):
    def __init__(self):
        super().__init__('jetson_perception_node')
        self.get_logger().info("Initialisation du Nœud de Perception IA...")

        # 1. Publishers
        # Image brute
        self.pub_image = self.create_publisher(Image, '/camera/image_raw', 10)
        # Image de Debug  (résultat)
        self.pub_debug = self.create_publisher(Image, '/perception/debug_view', 10)
        # Masque binaire de la route (pour Nav2)
        self.pub_drivable = self.create_publisher(Image, '/perception/drivable_area', 10)
        # Liste des détections
        self.pub_detections = self.create_publisher(Detection2DArray, '/perception/detections', 10)

        # Outil de conversion CV2 <-> ROS
        self.bridge = CvBridge()

        # 2. Initialisation CUDA / TensorRT
        self.device = torch.device('cuda:0')
        self.stream = torch.cuda.Stream()
        
        # Events pour mesurer la latence
        self.start_evt = torch.cuda.Event(enable_timing=True)
        self.end_evt = torch.cuda.Event(enable_timing=True)

        # Chargement des Modèles
        self.get_logger().info("--> Chargement YOLO...")
        self.yolo_model = YOLO(YOLO_ENGINE, task='detect')
        
        self.get_logger().info("--> Chargement TwinLiteNet...")
        self.trt_seg = TRTModuleStatic(TWINLITE_ENGINE, self.device)
        
        self.get_logger().info("--> Chargement Classifier...")
        self.trt_cls = TRTModuleStatic(CLASSIFIER_ENGINE, self.device)
        self.cls_resize = T.Resize((32, 32))

        # Warmup (Chauffer le GPU)
        self.do_warmup()

        # 3. Démarrage Caméra
        self.reader = USBCameraReader(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
        if not self.reader.working:
            self.get_logger().error("Echec critique: Caméra non détectée.")
            rclpy.shutdown()
            return

        # 4. Timer de boucle (30 Hz visé)
        self.timer = self.create_timer(0.01, self.inference_callback)
        self.get_logger().info("Système Prêt. Inférence active.")

    def do_warmup(self):
        """Exécute quelques passes à vide pour initialiser CUDA"""
        dummy_img = np.zeros((YOLO_IMG_SIZE, YOLO_IMG_SIZE, 3), dtype=np.uint8)
        for _ in range(5):
            self.yolo_model(dummy_img, imgsz=YOLO_IMG_SIZE, verbose=False)
        torch.cuda.synchronize()

    def inference_callback(self):
        """Boucle principale appelée par le Timer ROS"""
        frame = self.reader.read()
        if frame is None:
            return

        display_frame = frame.copy()
        timestamp = self.get_clock().now().to_msg()
        header_frame_id = "camera_link"

        # --- DÉBUT MESURE GPU ---
        self.start_evt.record(self.stream)

        with torch.cuda.stream(self.stream):
            # Préparation Tenseurs
            t_frame = torch.from_numpy(frame).to(self.device, non_blocking=True)
            # Normalisation 0-1 et channel first (HWC -> CHW)
            t_frame_float = t_frame.permute(2, 0, 1).unsqueeze(0).float().div(255.0)

            # YOLO Inférence
            results = self.yolo_model(frame, imgsz=YOLO_IMG_SIZE, conf=CONF_THRESH, verbose=False)[0]

            # TwinLiteNet Inférence
        
            t_seg_input = torch.nn.functional.interpolate(t_frame_float, size=(360, 640), mode='bilinear')
            seg_outs = self.trt_seg.forward(t_seg_input, self.stream)
            da_predict = seg_outs[0] # Drivable Area
            ll_predict = seg_outs[1] # Lane Lines

            # D. Préparation Classification (Feux Tricolores)
            batch_tensors = []
            traffic_light_boxes = [] # (x1, y1, x2, y2)
            
            boxes = results.boxes
            # Si on détecte des objets de classe 9 (Traffic Light dans COCO)
            if boxes.shape[0] > 0 and (boxes.cls == 9).any():
                light_boxes_coords = boxes.xyxy[boxes.cls == 9].cpu().numpy()
                for box in light_boxes_coords:
                    x1, y1, x2, y2 = map(int, box)
                    # Clamping pour ne pas sortir de l'image
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(self.reader.w, x2), min(self.reader.h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        # Crop du tenseur GPU directement
                        crop = t_frame_float[0, :, y1:y2, x1:x2]
                        # BGR -> RGB si nécessaire pour le classifieur
                        crop = crop[[2, 1, 0], :, :] 
                        crop = self.cls_resize(crop)
                        batch_tensors.append(crop)
                        traffic_light_boxes.append((x1, y1, x2, y2))
            
            # E. Inférence Classifieur (Batch)
            cls_results_map = {}
            if batch_tensors:
                batch_input = torch.stack(batch_tensors)
                # Normalisation spécifique au classifieur (souvent -0.5 / 0.5)
                batch_input = (batch_input - 0.5) / 0.5
                cls_out = self.trt_cls.forward(batch_input, self.stream)[0]
                # Softmax pour avoir les scores
                scores, idxs = torch.max(torch.softmax(cls_out, dim=1), 1)
                
                # Récupération CPU
                scores_cpu = scores.cpu().numpy()
                idxs_cpu = idxs.cpu().numpy()
                
                c_map = {1: 'Green', 2: 'Red', 3: 'Yellow'}
                for k in range(len(traffic_light_boxes)):
                    name = c_map.get(idxs_cpu[k], 'Unknown')
                    score = scores_cpu[k]
                    if name != 'Unknown' and score > CLS_CONF_THRESH:
                        cls_results_map[k] = (name, score)

        # --- FIN MESURE GPU ---
        self.end_evt.record(self.stream)
        self.stream.synchronize()
        latency_ms = self.start_evt.elapsed_time(self.end_evt)

        # ---------------------------------------------------------
        # --- PUBLICATION ET DESSIN ---
        # ---------------------------------------------------------

        # 1. Traitement TwinLiteNet (Masques)
        mask_da = da_predict[0, 1].cpu().numpy() # Channel 1 = Drivable
        mask_ll = ll_predict[0, 1].cpu().numpy()
        
        # Redimensionner le masque à la taille de l'image originale
        mask_da_resized = cv2.resize(mask_da, (self.reader.w, self.reader.h), interpolation=cv2.INTER_NEAREST)
        mask_ll_resized = cv2.resize(mask_ll, (self.reader.w, self.reader.h), interpolation=cv2.INTER_NEAREST)

        # Création Image ROS pour la zone roulable (Mono8 : 0=Obstacle, 255=Route)
        drivable_img = (mask_da_resized > 0.5).astype(np.uint8) * 255
        da_msg = self.bridge.cv2_to_imgmsg(drivable_img, encoding="mono8")
        da_msg.header.stamp = timestamp
        da_msg.header.frame_id = header_frame_id
        self.pub_drivable.publish(da_msg)

        # Dessin sur l'image de debug (Vert pour route, Bleu pour lignes)
        display_frame[mask_da_resized > 0.5] = (display_frame[mask_da_resized > 0.5] * 0.6 + np.array([0, 255, 0]) * 0.4).astype(np.uint8)
        display_frame[mask_ll_resized > 0.5] = (display_frame[mask_ll_resized > 0.5] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)

        # 2. Traitement Détections (YOLO + Classif)
        detect_array_msg = Detection2DArray()
        detect_array_msg.header.stamp = timestamp
        detect_array_msg.header.frame_id = header_frame_id

        light_counter = 0
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            label_text = ""
            color = (255, 255, 255)

            # Logique spécifique par classe
            if cls_id == 9: # Traffic Light
                if light_counter in cls_results_map:
                    color_name, score_cls = cls_results_map[light_counter]
                    label_text = f"{color_name} {score_cls:.2f}"
                    color = COLORS.get(color_name, (255, 255, 255))
                    
                    # On met à jour la classe pour ROS (ex: 'Red Light')
                    final_class_id = f"traffic_light_{color_name.lower()}"
                else:
                    final_class_id = "traffic_light_unknown"
                light_counter += 1
                
            elif cls_id == 0:
                label_text = f"Person {conf:.2f}"
                color = COLORS['Person']
                final_class_id = "person"
                
            elif cls_id == 11: # Stop Sign
                label_text = f"STOP {conf:.2f}"
                color = COLORS['Stop']
                final_class_id = "stop_sign"
                
            elif cls_id == 2: # Car
                label_text = f"Car {conf:.2f}"
                color = COLORS['Car']
                final_class_id = "car"
            else:
                continue # On ignore les autres classes pour l'instant

            # -- Création Message ROS Detection2D --
            detection = Detection2D()
            detection.header.stamp = timestamp
            detection.header.frame_id = header_frame_id
            
            # Bounding Box (Centre X, Centre Y, W, H)
            detection.bbox.center.position.x = float(x1 + x2) / 2
            detection.bbox.center.position.y = float(y1 + y2) / 2
            detection.bbox.size_x = float(x2 - x1)
            detection.bbox.size_y = float(y2 - y1)
            
            # Hypothesis
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = final_class_id
            hypothesis.hypothesis.score = conf
            detection.results.append(hypothesis)
            
            detect_array_msg.detections.append(detection)

            # Dessin Boîte Debug
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            if label_text:
                cv2.putText(display_frame, label_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Publication des détections
        self.pub_detections.publish(detect_array_msg)

        # 3. Publication Images Finales
        # HUD latence
        cv2.putText(display_frame, f"GPU: {latency_ms:.2f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Image debug
        debug_msg = self.bridge.cv2_to_imgmsg(display_frame, encoding="bgr8")
        debug_msg.header.stamp = timestamp
        debug_msg.header.frame_id = header_frame_id
        self.pub_debug.publish(debug_msg)

        # Image raw (optionnel, à désactiver si besoin de bande passante)
        # raw_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        # raw_msg.header.stamp = timestamp
        # raw_msg.header.frame_id = header_frame_id
        # self.pub_image.publish(raw_msg)
        
    def destroy_node(self):
        self.reader.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = JetsonPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
