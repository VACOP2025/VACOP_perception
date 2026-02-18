"""
Nœud ROS 2 de Perception : fusion YOLO26 + TwinLiteNet + Classifier.
Inclus : Envoi du flux traité (avec boîtes de détection) vers le Dashboard via GStreamer.
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
import paho.mqtt.client as mqtt ## -- dashboard --
import json                      ## -- dashboard --
import gc
import os

# --- CONFIGURATION CONSTANTES ---
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
BROKER_MQTT = "neocampus.univ-tlse3.fr" ## -- dashboard --

# Chemins des modèles
WORKSPACE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_ENGINE = os.path.join(WORKSPACE_PATH, "../models/yolo26m.engine")
TWINLITE_ENGINE = os.path.join(WORKSPACE_PATH, "../models/twinlite.engine")
CLASSIFIER_ENGINE = os.path.join(WORKSPACE_PATH, "../models/classifier.engine")

CONF_THRESH = 0.25
CLS_CONF_THRESH = 0.85
YOLO_IMG_SIZE = 320

COLORS = {
    'Red': (0, 0, 255), 'Green': (0, 255, 0), 'Yellow': (0, 255, 255),
    'Car': (100, 100, 100), 'Person': (255, 100, 0), 'Stop': (0, 0, 128)
}

# --- CLASSES UTILITAIRES ---

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
            self.working = False
            return
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

# --- NOEUD ROS 2 ---

class JetsonPerceptionNode(Node):
    def __init__(self):
        super().__init__('jetson_perception_node')
        
        # 1. Dashboard & MQTT Init ## -- dashboard --
        self.dashboard_ip = None ## -- dashboard --
        self.video_out = None ## -- dashboard --
        try: ## -- dashboard --
            self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2) ## -- dashboard --
            self.mqtt_client.on_message = self.on_mqtt_message ## -- dashboard --
            self.mqtt_client.connect(BROKER_MQTT, 1883) ## -- dashboard --
            self.mqtt_client.subscribe("TestTopic/VACOP/video/discovery") ## -- dashboard --
            threading.Thread(target=self.mqtt_client.loop_forever, daemon=True).start() ## -- dashboard --
        except Exception as e: ## -- dashboard --
            self.get_logger().error(f"MQTT Error: {e}") ## -- dashboard --

        # 2. ROS Publishers
        self.pub_image = self.create_publisher(Image, '/camera/image_raw', 10)
        self.pub_debug = self.create_publisher(Image, '/perception/debug_view', 10)
        self.pub_drivable = self.create_publisher(Image, '/perception/drivable_area', 10)
        self.pub_detections = self.create_publisher(Detection2DArray, '/perception/detections', 10)
        self.bridge = CvBridge()

        # 3. CUDA & Models
        self.device = torch.device('cuda:0')
        self.stream = torch.cuda.Stream()
        self.start_evt = torch.cuda.Event(enable_timing=True)
        self.end_evt = torch.cuda.Event(enable_timing=True)

        self.yolo_model = YOLO(YOLO_ENGINE, task='detect')
        self.trt_seg = TRTModuleStatic(TWINLITE_ENGINE, self.device)
        self.trt_cls = TRTModuleStatic(CLASSIFIER_ENGINE, self.device)
        self.cls_resize = T.Resize((32, 32))

        # 4. Camera Start
        self.reader = USBCameraReader(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
        self.timer = self.create_timer(0.01, self.inference_callback)

    def on_mqtt_message(self, client, userdata, msg): ## -- dashboard --
        try: ## -- dashboard --
            data = json.loads(msg.payload.decode()) ## -- dashboard --
            if "ip" in data and self.dashboard_ip is None: ## -- dashboard --
                self.dashboard_ip = data["ip"] ## -- dashboard --
                self.init_gstreamer_sink() ## -- dashboard --
        except Exception as e: ## -- dashboard --
            self.get_logger().error(f"MQTT Parse Error: {e}") ## -- dashboard --

    def init_gstreamer_sink(self): ## -- dashboard --
        gst_pipeline = ( ## -- dashboard --
            f"appsrc ! videoconvert ! video/x-raw, format=I420 ! " ## -- dashboard --
            f"v4l2h264enc bitrate=3000000 ! mpegtsmux ! " ## -- dashboard --
            f"udpsink host={self.dashboard_ip} port=5001" ## -- dashboard --
        ) ## -- dashboard --
        self.video_out = cv2.VideoWriter(gst_pipeline, cv2.CAP_GSTREAMER, 0, 30.0, (CAMERA_WIDTH, CAMERA_HEIGHT)) ## -- dashboard --

    def inference_callback(self):
        frame = self.reader.read()
        if frame is None: return

        display_frame = frame.copy()
        timestamp = self.get_clock().now().to_msg()
        header_frame_id = "camera_link"

        self.start_evt.record(self.stream)
        with torch.cuda.stream(self.stream):
            t_frame = torch.from_numpy(frame).to(self.device, non_blocking=True)
            t_frame_float = t_frame.permute(2, 0, 1).unsqueeze(0).float().div(255.0)

            # Inférences
            results = self.yolo_model(frame, imgsz=YOLO_IMG_SIZE, conf=CONF_THRESH, verbose=False)[0]
            t_seg_input = torch.nn.functional.interpolate(t_frame_float, size=(360, 640), mode='bilinear')
            seg_outs = self.trt_seg.forward(t_seg_input, self.stream)
            
            # (Ici ton code de classification Traffic Light...)
            cls_results_map = {} # Rempli par ta logique de classification

        self.end_evt.record(self.stream)
        self.stream.synchronize()
        latency_ms = self.start_evt.elapsed_time(self.end_evt)

        # 1. TwinLiteNet Drawing
        mask_da = seg_outs[0][0, 1].cpu().numpy()
        mask_da_resized = cv2.resize(mask_da, (self.reader.w, self.reader.h))
        display_frame[mask_da_resized > 0.5] = (display_frame[mask_da_resized > 0.5] * 0.6 + np.array([0, 255, 0]) * 0.4).astype(np.uint8)

        # 2. YOLO Drawing & ROS Msg
        detect_array_msg = Detection2DArray()
        detect_array_msg.header.stamp = timestamp
        detect_array_msg.header.frame_id = header_frame_id

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # Dessin
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            
            # ROS Message
            detection = Detection2D()
            detection.bbox.center.position.x = float(x1 + x2) / 2
            detection.bbox.center.position.y = float(y1 + y2) / 2
            detection.bbox.size_x = float(x2 - x1)
            detection.bbox.size_y = float(y2 - y1)
            detect_array_msg.detections.append(detection)

        # 3. HUD & Publication
        cv2.putText(display_frame, f"GPU: {latency_ms:.2f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # --- ENVOI DASHBOARD --- ## -- dashboard --
        if self.video_out is not None: ## -- dashboard --
            self.video_out.write(display_frame) ## -- dashboard --

        # ROS Debug Image
        debug_msg = self.bridge.cv2_to_imgmsg(display_frame, encoding="bgr8")
        debug_msg.header.stamp = timestamp
        debug_msg.header.frame_id = header_frame_id
        self.pub_debug.publish(debug_msg)
        self.pub_detections.publish(detect_array_msg)

    def destroy_node(self):
        if self.video_out: self.video_out.release() ## -- dashboard --
        self.mqtt_client.disconnect() ## -- dashboard --
        self.reader.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = JetsonPerceptionNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
