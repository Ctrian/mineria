import cv2
import torch
from ultralytics import YOLO
from reid_utils import load_reid_model, get_embedding, get_dominant_color
import numpy as np
from datetime import datetime
import threading
import time
import json
import sqlite3
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import queue
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

@dataclass
class GridZone:
    id: str
    name: str
    x1: int
    y1: int
    x2: int
    y2: int
    color: Tuple[int, int, int] = (0, 255, 255)

@dataclass
class PersonEvent:
    person_id: int
    timestamp: str
    event_type: str  # 'entry', 'zone_enter', 'zone_exit', 'exit'
    zone_id: Optional[str] = None
    coordinates: Optional[Tuple[int, int]] = None
    confidence: float = 0.0

class MultiEmbeddingPerson:
    """Clase para manejar múltiples embeddings por persona"""
    def __init__(self, person_id: int):
        self.id = person_id
        self.embeddings = []
        self.dominant_colors = []
        self.poses = []
        self.timestamps = []
        self.detection_count = 0
        self.quality_scores = []
        self.max_embeddings = 15
        
    def add_embedding(self, embedding: np.ndarray, color: List[int], 
                     pose: str = None, quality_score: float = 1.0):
        self.embeddings.append(embedding)
        self.dominant_colors.append(color)
        self.poses.append(pose or 'unknown')
        self.timestamps.append(datetime.now())
        self.quality_scores.append(quality_score)
        self.detection_count += 1
        
        if len(self.embeddings) > self.max_embeddings:
            self._prune_embeddings()
    
    def _prune_embeddings(self):
        if len(self.embeddings) <= self.max_embeddings:
            return
        embeddings_array = np.array(self.embeddings)
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
        clusters = clustering.fit_predict(embeddings_array)
        
        keep_indices = []
        for cluster_id in set(clusters):
            if cluster_id == -1:
                outlier_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
                keep_indices.extend(outlier_indices)
            else:
                cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
                best_idx = max(cluster_indices, key=lambda i: self.quality_scores[i])
                keep_indices.append(best_idx)
        
        if len(keep_indices) > self.max_embeddings:
            keep_indices.sort(key=lambda i: (self.quality_scores[i], self.timestamps[i]), reverse=True)
            keep_indices = keep_indices[:self.max_embeddings]
        
        self.embeddings = [self.embeddings[i] for i in keep_indices]
        self.dominant_colors = [self.dominant_colors[i] for i in keep_indices]
        self.poses = [self.poses[i] for i in keep_indices]
        self.timestamps = [self.timestamps[i] for i in keep_indices]
        self.quality_scores = [self.quality_scores[i] for i in keep_indices]
    
    def find_best_match(self, query_embedding: np.ndarray, threshold: float = 0.65) -> Tuple[bool, float, str]:
        if not self.embeddings:
            return False, 0.0, "no_embeddings"
        
        similarities = [cosine_similarity(query_embedding.reshape(1, -1), e.reshape(1, -1))[0][0] for e in self.embeddings]
        
        best_similarity = max(similarities)
        best_idx = similarities.index(best_similarity)
        best_pose = self.poses[best_idx]
        
        dynamic_threshold = threshold * (0.95 if len(self.embeddings) >= 5 else 1.05)
        is_match = best_similarity > dynamic_threshold
        
        return is_match, best_similarity, best_pose

class EnhancedDatabaseManager:
    """Gestor de base de datos mejorado para multi-embedding."""
    def __init__(self, db_path='enhanced_tracking_data.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY, first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP, total_visits INTEGER DEFAULT 1,
                detection_count INTEGER DEFAULT 1, avg_confidence REAL DEFAULT 0.0,
                dominant_color TEXT, pose_distribution TEXT
            )''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS person_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT, person_id INTEGER, embedding BLOB,
                pose TEXT, quality_score REAL, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES persons (id)
            )''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tracking_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, person_id INTEGER, timestamp TIMESTAMP,
                event_type TEXT, zone_id TEXT, x_coordinate INTEGER, y_coordinate INTEGER,
                confidence REAL, similarity_score REAL, matched_pose TEXT,
                FOREIGN KEY (person_id) REFERENCES persons (id)
            )''')
        conn.commit()
        conn.close()
    
    def save_person(self, person: MultiEmbeddingPerson):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        avg_color = np.mean(person.dominant_colors, axis=0).astype(int).tolist() if person.dominant_colors else [128,128,128]
        pose_dist = defaultdict(int)
        for pose in person.poses: pose_dist[pose] += 1
        
        cursor.execute('INSERT OR REPLACE INTO persons (id, detection_count, dominant_color, pose_distribution, last_seen) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)', 
                       (person.id, person.detection_count, json.dumps(avg_color), json.dumps(pose_dist)))
        
        cursor.execute('DELETE FROM person_embeddings WHERE person_id = ?', (person.id,))
        for i, embedding in enumerate(person.embeddings):
            cursor.execute('INSERT INTO person_embeddings (person_id, embedding, pose, quality_score, timestamp) VALUES (?, ?, ?, ?, ?)',
                           (person.id, embedding.tobytes(), person.poses[i], person.quality_scores[i], person.timestamps[i]))
        conn.commit()
        conn.close()

    def load_person(self, person_id: int) -> Optional[MultiEmbeddingPerson]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM persons WHERE id = ?', (person_id,))
        if not cursor.fetchone():
            conn.close()
            return None
        
        person = MultiEmbeddingPerson(person_id)
        cursor.execute('SELECT embedding, pose, quality_score, timestamp FROM person_embeddings WHERE person_id = ? ORDER BY timestamp', (person_id,))
        for emb_data in cursor.fetchall():
            person.embeddings.append(np.frombuffer(emb_data[0], dtype=np.float32))
            person.poses.append(emb_data[1])
            person.quality_scores.append(emb_data[2])
            person.timestamps.append(datetime.fromisoformat(emb_data[3]))
        conn.close()
        return person

    def insert_event(self, event: PersonEvent, similarity_score: float = 0.0, matched_pose: str = ""):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        coords = event.coordinates or (None, None)
        cursor.execute('INSERT INTO tracking_events (person_id, timestamp, event_type, zone_id, x_coordinate, y_coordinate, confidence, similarity_score, matched_pose) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                       (event.person_id, event.timestamp, event.event_type, event.zone_id, coords[0], coords[1], event.confidence, similarity_score, matched_pose))
        conn.commit()
        conn.close()

class GridManager:
    def __init__(self, frame_width: int, frame_height: int):
        self.zones: List[GridZone] = []
        grid_cols, grid_rows = 3, 3
        col_w, row_h = frame_width // grid_cols, frame_height // grid_rows
        colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(128,0,128),(255,165,0),(0,128,128)]
        for r in range(grid_rows):
            for c in range(grid_cols):
                self.zones.append(GridZone(f"S{r}_{c}", f"Stand {r+1}-{c+1}", c*col_w, r*row_h, (c+1)*col_w, (r+1)*row_h, colors[len(self.zones)%len(colors)]))

    def get_zone_for_point(self, x: int, y: int) -> Optional[GridZone]:
        for zone in self.zones:
            if zone.x1 <= x <= zone.x2 and zone.y1 <= y <= zone.y2: return zone
        return None

    def draw_grid(self, frame: np.ndarray):
        for z in self.zones:
            cv2.rectangle(frame, (z.x1, z.y1), (z.x2, z.y2), z.color, 2)
            cv2.putText(frame, z.name, (z.x1 + 5, z.y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, z.color, 1)

class PoseEstimator:
    @staticmethod
    def estimate_pose(w: int, h: int) -> str:
        ratio = w / h if h > 0 else 1
        if ratio > 0.7: return "side"
        elif ratio < 0.4: return "front_back"
        else: return "diagonal"

    @staticmethod
    def calculate_quality_score(conf: float, area: int) -> float:
        return min((conf * 0.6 + min(area / 5000.0, 1.0) * 0.4), 1.0)

class EnhancedTrackingSystem:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Usando dispositivo: {self.device}")
        self.yolo = YOLO("yolov8n.pt")
        self.reid_model = load_reid_model()
        self.db_manager = EnhancedDatabaseManager()
        self.grid_manager: Optional[GridManager] = None
        self.pose_estimator = PoseEstimator()
        self.persons_db: Dict[int, MultiEmbeddingPerson] = {}
        self.current_max_id = 0
        self.trajectories = defaultdict(list)
        self.person_zones = defaultdict(str)
        self.db_lock = threading.Lock()
        self.event_queue = queue.Queue()
        self.event_processor_thread = threading.Thread(target=self._process_events, daemon=True)
        self.event_processor_thread.start()
        self._load_existing_persons()

    def _load_existing_persons(self):
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM persons')
        for (pid,) in cursor.fetchall():
            person = self.db_manager.load_person(pid)
            if person:
                self.persons_db[pid] = person
                self.current_max_id = max(self.current_max_id, pid)
        conn.close()
        print(f"[INFO] Cargadas {len(self.persons_db)} personas de la base de datos.")

    def _process_events(self):
        while True:
            try:
                event, sim, pose = self.event_queue.get(timeout=1)
                self.db_manager.insert_event(event, sim, pose)
                print(f"[EVENT] {event.event_type}: Persona {event.person_id} -> {event.zone_id or 'N/A'}")
                self.event_queue.task_done()
            except queue.Empty:
                continue

    def _get_person_details(self, crop, box):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        embedding = get_embedding(self.reid_model, crop)
        pose = self.pose_estimator.estimate_pose(w, h)
        quality = self.pose_estimator.calculate_quality_score(float(box.conf[0]), w * h)
        return embedding, pose, quality

    def _identify_or_create_person(self, embedding: np.ndarray, crop: np.ndarray, pose: str, quality: float) -> Tuple[int, float]:
        """Usado por CAM1: Encuentra la mejor coincidencia o crea una nueva persona."""
        best_match_id, best_sim = None, 0.0
        with self.db_lock:
            for pid, person in self.persons_db.items():
                is_match, sim, _ = person.find_best_match(embedding)
                if is_match and sim > best_sim:
                    best_match_id, best_sim = pid, sim
            
            if best_match_id is None:
                self.current_max_id += 1
                new_id = self.current_max_id
                new_person = MultiEmbeddingPerson(new_id)
                new_person.add_embedding(embedding, get_dominant_color(crop), pose, quality)
                self.persons_db[new_id] = new_person
                self.db_manager.save_person(new_person)
                return new_id, 1.0  # 1.0 para indicar que es nuevo
            else:
                person = self.persons_db[best_match_id]
                person.add_embedding(embedding, get_dominant_color(crop), pose, quality)
                if person.detection_count % 5 == 0:
                    self.db_manager.save_person(person)
                return best_match_id, best_sim

    def _reidentify_person(self, embedding: np.ndarray) -> Tuple[Optional[int], float, str]:
        """Usado por CAM2: Solo busca coincidencias, no crea nuevas."""
        best_match_id, best_sim, best_pose = None, 0.0, "unknown"
        # Umbral más estricto para la re-identificación
        REID_THRESHOLD = 0.7 
        with self.db_lock:
            for pid, person in self.persons_db.items():
                is_match, sim, pose = person.find_best_match(embedding, threshold=REID_THRESHOLD)
                if is_match and sim > best_sim:
                    best_match_id, best_sim, best_pose = pid, sim, pose
        return best_match_id, best_sim, best_pose

    def camera_1_entry(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[ERROR] No se pudo abrir la cámara {camera_index}")
            return
        print(f"[INFO] Cámara 1 (Entrada) iniciada.")

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            results = self.yolo(frame)[0]
            for box in results.boxes:
                if int(box.cls[0]) != 0: continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue
                
                embedding, pose, quality = self._get_person_details(crop, box)
                person_id, sim = self._identify_or_create_person(embedding, crop, pose, quality)
                
                color = (0, 255, 0) if sim >= 0.99 else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID: {person_id}" + (" (Nuevo)" if sim >= 0.99 else f" (Sim: {sim:.2f})")
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.putText(frame, "CAMARA 1 - ENTRADA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Cámara 1 - Entrada", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()

    def camera_2_tracking(self, camera_index=1):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[ERROR] No se pudo abrir la cámara {camera_index}")
            return
        print(f"[INFO] Cámara 2 (Tracking) iniciada.")

        while True:
            ret, frame = cap.read()
            if not ret: break
            
            if self.grid_manager is None and frame.shape[0] > 0:
                self.grid_manager = GridManager(frame.shape[1], frame.shape[0])
            if self.grid_manager:
                self.grid_manager.draw_grid(frame)

            results = self.yolo(frame)[0]
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            for box in results.boxes:
                if int(box.cls[0]) != 0: continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue

                embedding, _, _ = self._get_person_details(crop, box)
                person_id, sim, pose = self._reidentify_person(embedding)
                
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                if person_id is not None:
                    # --- Persona Identificada ---
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    person_info = self.persons_db.get(person_id)
                    emb_count = len(person_info.embeddings) if person_info else 0
                    
                    cv2.putText(frame, f"ID: {person_id} ({emb_count} emb)", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(frame, f"Sim: {sim:.2f} Pose: {pose}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    # Lógica de zonas
                    zone = self.grid_manager.get_zone_for_point(*center) if self.grid_manager else None
                    prev_zone_id = self.person_zones.get(person_id)
                    if zone and zone.id != prev_zone_id:
                        if prev_zone_id:
                            self.event_queue.put((PersonEvent(person_id, current_time, 'zone_exit', prev_zone_id, center, float(box.conf[0])), sim, pose))
                        self.event_queue.put((PersonEvent(person_id, current_time, 'zone_enter', zone.id, center, float(box.conf[0])), sim, pose))
                        self.person_zones[person_id] = zone.id
                else:
                    # --- Persona Desconocida ---
                    color = (0, 0, 255) # Rojo
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, "ID: -1", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(frame, "Persona Desconocida", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.putText(frame, "CAMARA 2 - TRACKING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Cámara 2 - Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()

if __name__ == "__main__":
    # --- Punto de entrada principal ---
    tracker_system = EnhancedTrackingSystem()
    
    # Crear e iniciar hilos para cada cámara
    # Usa 0 para la cámara principal, 1 para la secundaria, etc.
    # Puedes usar rutas a archivos de video como "path/to/video1.mp4"
    thread_cam1 = threading.Thread(target=tracker_system.camera_1_entry, args=(0,))
    thread_cam2 = threading.Thread(target=tracker_system.camera_2_tracking, args=(1,))
    
    print("[INFO] Iniciando hilos de cámaras...")
    thread_cam1.start()
    thread_cam2.start()
    
    # Esperar a que los hilos terminen (en este caso, nunca, hasta que se presione 'q')
    thread_cam1.join()
    thread_cam2.join()
    
    # Limpieza final
    print("[INFO] Cerrando aplicación.")
    cv2.destroyAllWindows()
