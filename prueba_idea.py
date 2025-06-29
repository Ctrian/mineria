import cv2
import torch
from ultralytics import YOLO
from reid_utils import load_reid_model, get_embedding, load_database, save_database, is_same_person, get_dominant_color
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

class DatabaseManager:
    def __init__(self, db_path='tracking_data.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Inicializa las tablas de la base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabla de personas
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_visits INTEGER DEFAULT 1,
                embedding BLOB,
                dominant_color TEXT
            )
        ''')
        
        # Tabla de eventos de tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tracking_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                timestamp TIMESTAMP,
                event_type TEXT,
                zone_id TEXT,
                x_coordinate INTEGER,
                y_coordinate INTEGER,
                confidence REAL,
                FOREIGN KEY (person_id) REFERENCES persons (id)
            )
        ''')
        
        # Tabla de zonas/stands
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS zones (
                id TEXT PRIMARY KEY,
                name TEXT,
                x1 INTEGER,
                y1 INTEGER,
                x2 INTEGER,
                y2 INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def insert_person(self, person_id: int, embedding: np.ndarray, dominant_color: List[int]):
        """Inserta o actualiza una persona en la base de datos"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        embedding_blob = embedding.tobytes()
        color_json = json.dumps(dominant_color)
        
        cursor.execute('''
            INSERT OR REPLACE INTO persons (id, embedding, dominant_color, last_seen)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ''', (person_id, embedding_blob, color_json))
        
        conn.commit()
        conn.close()
    
    def insert_event(self, event: PersonEvent):
        """Inserta un evento de tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        x_coord = event.coordinates[0] if event.coordinates else None
        y_coord = event.coordinates[1] if event.coordinates else None
        
        cursor.execute('''
            INSERT INTO tracking_events 
            (person_id, timestamp, event_type, zone_id, x_coordinate, y_coordinate, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (event.person_id, event.timestamp, event.event_type, 
              event.zone_id, x_coord, y_coord, event.confidence))
        
        conn.commit()
        conn.close()
    
    def get_person_stats(self, person_id: int) -> Dict:
        """Obtiene estadísticas de una persona"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) as total_events,
                   COUNT(CASE WHEN event_type = 'zone_enter' THEN 1 END) as zones_visited,
                   MIN(timestamp) as first_seen,
                   MAX(timestamp) as last_seen
            FROM tracking_events 
            WHERE person_id = ?
        ''', (person_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        return {
            'total_events': result[0],
            'zones_visited': result[1],
            'first_seen': result[2],
            'last_seen': result[3]
        }

class GridManager:
    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.zones: List[GridZone] = []
        self.setup_default_grid()
    
    def setup_default_grid(self):
        """Configura un grid por defecto 3x3"""
        grid_cols = 3
        grid_rows = 3
        
        col_width = self.frame_width // grid_cols
        row_height = self.frame_height // grid_rows
        
        for row in range(grid_rows):
            for col in range(grid_cols):
                zone_id = f"S{row}_{col}"  # Stand row_col
                zone_name = f"Stand {row+1}-{col+1}"
                
                x1 = col * col_width
                y1 = row * row_height
                x2 = x1 + col_width
                y2 = y1 + row_height
                
                # Colores diferentes para cada zona
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
                         (255, 255, 0), (255, 0, 255), (0, 255, 255),
                         (128, 0, 128), (255, 165, 0), (0, 128, 128)]
                
                color = colors[len(self.zones) % len(colors)]
                
                self.zones.append(GridZone(zone_id, zone_name, x1, y1, x2, y2, color))
    
    def get_zone_for_point(self, x: int, y: int) -> Optional[GridZone]:
        """Obtiene la zona que contiene un punto dado"""
        for zone in self.zones:
            if zone.x1 <= x <= zone.x2 and zone.y1 <= y <= zone.y2:
                return zone
        return None
    
    def draw_grid(self, frame: np.ndarray, show_labels: bool = True):
        """Dibuja el grid en el frame"""
        for zone in self.zones:
            # Dibujar rectángulo de la zona
            cv2.rectangle(frame, (zone.x1, zone.y1), (zone.x2, zone.y2), zone.color, 2)
            
            if show_labels:
                # Dibujar etiqueta de la zona
                cv2.putText(frame, zone.name, 
                           (zone.x1 + 5, zone.y1 + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone.color, 1)

class ImprovedTrackingSystem:
    def __init__(self):
        # Inicialización de componentes
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] Usando dispositivo: {self.device}")
        
        # Modelos
        self.yolo = YOLO("yolov8n.pt")
        self.reid_model = load_reid_model()
        self.db = load_database()
        
        # Managers
        self.db_manager = DatabaseManager()
        self.grid_manager = None  # Se inicializa cuando se conoce el tamaño del frame
        
        # Variables de estado
        self.current_max_id = max([p['id'] for p in self.db], default=0)
        self.trajectories = defaultdict(list)
        self.person_zones = defaultdict(str)  # person_id -> current_zone_id
        self.db_lock = threading.Lock()
        
        # Cola de eventos para procesar en tiempo real
        self.event_queue = queue.Queue()
        
        # Hilos de procesamiento
        self.event_processor_thread = threading.Thread(target=self._process_events, daemon=True)
        self.event_processor_thread.start()
    
    def _process_events(self):
        """Procesa eventos de la cola en tiempo real"""
        while True:
            try:
                event = self.event_queue.get(timeout=1)
                self.db_manager.insert_event(event)
                print(f"[EVENT] {event.event_type}: Persona {event.person_id} -> {event.zone_id or 'N/A'}")
                self.event_queue.task_done()
            except queue.Empty:
                continue
    
    def find_person_in_database(self, embedding: np.ndarray, threshold: float = 0.75) -> int:
        """
        Busca una persona en la base de datos usando su embedding.
        Retorna el ID de la persona si se encuentra, -1 si no existe.
        """
        with self.db_lock:
            for entry in self.db:
                if is_same_person(entry['embedding'], embedding, threshold=threshold):
                    return entry['id']
        return -1  # Persona desconocida
    
    def camera_1_entry(self, camera_index=0):
        """Cámara 1: Registro de nuevas personas con detección de entrada"""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[ERROR] No se pudo abrir la cámara {camera_index}")
            return
        
        print(f"[INFO] Cámara 1 (Entrada) iniciada en índice {camera_index}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Inicializar grid manager si no existe
            if self.grid_manager is None:
                self.grid_manager = GridManager(frame.shape[1], frame.shape[0])
            
            results = self.yolo(frame)[0]
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            for box in results.boxes:
                if int(box.cls[0]) != 0:  # solo personas
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                confidence = float(box.conf[0])

                if crop.size == 0:
                    continue

                embedding = get_embedding(self.reid_model, crop)
                color = get_dominant_color(crop)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                match_id = None
                
                with self.db_lock:
                    for entry in self.db:
                        if is_same_person(entry['embedding'], embedding, threshold=0.75):
                            match_id = entry['id']
                            break

                    if match_id is None:
                        self.current_max_id += 1
                        match_id = self.current_max_id
                        self.db.append({'id': match_id, 'embedding': embedding, 'dominant_color': color})
                        save_database(self.db)
                        
                        # Guardar en base de datos SQL
                        self.db_manager.insert_person(match_id, embedding, color)
                        
                        # Evento de entrada
                        entry_event = PersonEvent(
                            person_id=match_id,
                            timestamp=current_time,
                            event_type='entry',
                            coordinates=center,
                            confidence=confidence
                        )
                        self.event_queue.put(entry_event)
                        
                        print(f"[CAM1-NUEVO] Persona registrada con ID {match_id}")

                # Dibujar detección
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {match_id}", (x1, y1 - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Conf: {confidence:.2f}", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Información en pantalla
            cv2.putText(frame, "CAMARA 1 - ENTRADA", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Personas: {len(self.db)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow("Cámara 1 - Entrada", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
    
    def camera_2_tracking(self, camera_index=1):
        """Cámara 2: Tracking con grid de zonas - SOLO BÚSQUEDA, NO REGISTRO"""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[ERROR] No se pudo abrir la cámara {camera_index}")
            return
        
        print(f"[INFO] Cámara 2 (Tracking) iniciada en índice {camera_index}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Inicializar grid manager si no existe
            if self.grid_manager is None:
                self.grid_manager = GridManager(frame.shape[1], frame.shape[0])
            
            # Dibujar grid
            self.grid_manager.draw_grid(frame)
            
            results = self.yolo(frame)[0]
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            for box in results.boxes:
                if int(box.cls[0]) != 0:  # solo personas
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                confidence = float(box.conf[0])
                
                if crop.size == 0:
                    continue

                embedding = get_embedding(self.reid_model, crop)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                # SOLO BUSCAR en la base de datos existente - NO REGISTRAR nuevas personas
                match_id = self.find_person_in_database(embedding, threshold=0.75)

                if match_id != -1:  # Persona conocida encontrada
                    # Actualizar trayectoria
                    self.trajectories[match_id].append(center)
                    if len(self.trajectories[match_id]) > 30:
                        self.trajectories[match_id] = self.trajectories[match_id][-30:]
                    
                    # Detectar zona actual
                    current_zone = self.grid_manager.get_zone_for_point(center[0], center[1])
                    previous_zone_id = self.person_zones.get(match_id)
                    
                    if current_zone:
                        current_zone_id = current_zone.id
                        
                        # Si cambió de zona
                        if previous_zone_id != current_zone_id:
                            # Evento de salida de zona anterior
                            if previous_zone_id:
                                exit_event = PersonEvent(
                                    person_id=match_id,
                                    timestamp=current_time,
                                    event_type='zone_exit',
                                    zone_id=previous_zone_id,
                                    coordinates=center,
                                    confidence=confidence
                                )
                                self.event_queue.put(exit_event)
                            
                            # Evento de entrada a nueva zona
                            enter_event = PersonEvent(
                                person_id=match_id,
                                timestamp=current_time,
                                event_type='zone_enter',
                                zone_id=current_zone_id,
                                coordinates=center,
                                confidence=confidence
                            )
                            self.event_queue.put(enter_event)
                            
                            self.person_zones[match_id] = current_zone_id
                    
                    # Dibujar detección - PERSONA CONOCIDA (Verde)
                    color_rect = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color_rect, 2)
                    cv2.putText(frame, f"ID: {match_id}", (x1, y1 - 45),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_rect, 2)
                    cv2.putText(frame, f"Zona: {current_zone.name if current_zone else 'N/A'}", 
                               (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_rect, 1)
                    cv2.putText(frame, f"Conf: {confidence:.2f}", (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    print(f"[CAM2-TRACKING] Persona ID {match_id} en zona {current_zone.name if current_zone else 'N/A'}")
                    
                else:  # match_id == -1: Persona desconocida
                    # Dibujar detección - PERSONA DESCONOCIDA (Rojo)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "ID: -1 (DESCONOCIDO)", (x1, y1 - 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(frame, f"Conf: {confidence:.2f}", (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    print(f"[CAM2-UNKNOWN] Persona desconocida detectada - ID: -1")

            # Dibujar trayectorias solo para personas conocidas
            for track_id, points in self.trajectories.items():
                if len(points) > 1:
                    for i in range(1, len(points)):
                        cv2.line(frame, points[i-1], points[i], (255, 0, 255), 2)

            # Información en pantalla
            cv2.putText(frame, "CAMARA 2 - TRACKING + GRID", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(frame, f"Tracking: {len(self.trajectories)} personas conocidas", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            cv2.putText(frame, f"DB Total: {len(self.db)} personas registradas", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            cv2.imshow("Cámara 2 - Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
    
    def run(self):
        """Ejecuta el sistema completo"""
        print("[INFO] Iniciando sistema mejorado de tracking...")
        
        # Verificar cámaras disponibles
        available_cameras = []
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        
        if len(available_cameras) < 2:
            print("[ERROR] Se necesitan al menos 2 cámaras")
            return
        
        print(f"[INFO] Cámaras disponibles: {available_cameras}")
        print(f"[INFO] Cámara 1 (Entrada): {available_cameras[0]}")
        print(f"[INFO] Cámara 2 (Tracking): {available_cameras[1]}")
        
        # Crear threads
        thread1 = threading.Thread(target=self.camera_1_entry, args=(available_cameras[0],))
        thread2 = threading.Thread(target=self.camera_2_tracking, args=(available_cameras[1],))
        
        thread1.daemon = True
        thread2.daemon = True
        thread1.start()
        thread2.start()
        
        try:
            while thread1.is_alive() and thread2.is_alive():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[INFO] Deteniendo sistema...")
        
        cv2.destroyAllWindows()

def main():
    system = ImprovedTrackingSystem()
    system.run()

if __name__ == "__main__":
    main()