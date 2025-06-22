import cv2
import torch
from ultralytics import YOLO
from reid_utils import load_reid_model, get_embedding, load_database, save_database, is_same_person, get_dominant_color
import numpy as np
from datetime import datetime
import threading
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Usando dispositivo: {device}")

# Cargar modelos (compartidos entre cámaras)
yolo = YOLO("yolov8n.pt")
reid_model = load_reid_model()
db = load_database()

# Variables globales compartidas
current_max_id = max([p['id'] for p in db], default=0)
trajectories = {}
db_lock = threading.Lock()  # Para acceso seguro a la base de datos

def color_distance(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))

def distancia_puntos(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def camera_1_entry(camera_index=0):
    """Cámara 1: Registro de nuevas personas"""
    global current_max_id, db
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara {camera_index}")
        return
    
    print(f"[INFO] Cámara 1 (Entrada) iniciada en índice {camera_index}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] No se pudo leer el frame de la cámara 1")
            break
            
        results = yolo(frame)[0]
        current_time = datetime.now().strftime("%H:%M:%S")

        for box in results.boxes:
            if int(box.cls[0]) != 0:  # solo personas
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:  # Verificar que el crop no esté vacío
                continue

            embedding = get_embedding(reid_model, crop)
            color = get_dominant_color(crop)

            match_id = None
            
            # Acceso seguro a la base de datos
            with db_lock:
                for entry in db:
                    if is_same_person(entry['embedding'], embedding, threshold=0.75):
                        match_id = entry['id']
                        break

                if match_id is None:
                    current_max_id += 1
                    match_id = current_max_id
                    db.append({'id': match_id, 'embedding': embedding, 'dominant_color': color})
                    save_database(db)
                    print(f"[CAM1-NUEVO] Persona registrada con ID {match_id}")
                else:
                    print(f"[CAM1-INFO] Persona conocida con ID {match_id}")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {match_id}", (x1, y1 - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"CAM1 - {current_time}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Agregar información en la esquina
        cv2.putText(frame, "CAMARA 1 - ENTRADA", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Personas registradas: {len(db)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("Cámara 1 - Entrada", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

def camera_2_tracking(camera_index=1):
    """Cámara 2: Tracking y verificación de IDs"""
    global db, trajectories
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara {camera_index}")
        return
    
    print(f"[INFO] Cámara 2 (Tracking) iniciada en índice {camera_index}")
    
    # Parámetros de tracking
    color_umbral = 50
    embedding_umbral = 0.75
    distancia_espacial_umbral = 100

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] No se pudo leer el frame de la cámara 2")
            break

        results = yolo(frame)[0]
        current_time = datetime.now().strftime("%H:%M:%S")

        for box in results.boxes:
            if int(box.cls[0]) != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue

            embedding = get_embedding(reid_model, crop)
            color = get_dominant_color(crop)
            centro = ((x1 + x2) // 2, (y1 + y2) // 2)

            match_id = None
            min_score = float('inf')
            
            # Acceso seguro a la base de datos
            with db_lock:
                for entry in db:
                    if is_same_person(entry['embedding'], embedding, threshold=embedding_umbral):
                        dist_c = color_distance(entry.get('dominant_color', [0, 0, 0]), color)
                        if dist_c > color_umbral:
                            continue

                        # Verificar distancia espacial
                        last_pos = trajectories.get(entry['id'], [])
                        if last_pos:
                            dist_pos = distancia_puntos(last_pos[-1], centro)
                            if dist_pos > distancia_espacial_umbral:
                                continue
                        else:
                            dist_pos = 0

                        score = dist_c + dist_pos
                        if score < min_score:
                            min_score = score
                            match_id = entry['id']

            # Actualizar trayectoria
            if match_id is not None and match_id != -1:
                if match_id not in trajectories:
                    trajectories[match_id] = []
                trajectories[match_id].append(centro)
                # Mantener solo los últimos 20 puntos
                if len(trajectories[match_id]) > 20:
                    trajectories[match_id] = trajectories[match_id][-20:]
                    
                print(f"[CAM2-TRACK] ID {match_id} detectado en posición {centro}")
            else:
                match_id = -1
                print(f"[CAM2-UNKNOWN] Persona desconocida detectada")

            # Dibujar detección
            color_rect = (0, 255, 0) if match_id != -1 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_rect, 2)
            cv2.putText(frame, f"ID: {match_id}", (x1, y1 - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_rect, 2)
            cv2.putText(frame, f"CAM2 - {current_time}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Dibujar trayectorias
        for track_id, points in trajectories.items():
            if len(points) > 1:
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], (255, 0, 255), 2)

        # Información en pantalla
        cv2.putText(frame, "CAMARA 2 - TRACKING", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(frame, f"Tracking {len(trajectories)} personas", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        cv2.imshow("Cámara 2 - Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

def main():
    print("[INFO] Iniciando sistema de doble cámara...")
    print("[INFO] Presiona 'q' en cualquier ventana para salir")
    
    # Verificar cámaras disponibles
    available_cameras = []
    for i in range(4):  # Verificar hasta 4 cámaras
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    
    print(f"[INFO] Cámaras disponibles: {available_cameras}")
    
    if len(available_cameras) < 2:
        print("[ERROR] Se necesitan al menos 2 cámaras para este sistema")
        return
    
    # Usar las dos primeras cámaras disponibles
    cam1_index = available_cameras[0]
    cam2_index = available_cameras[1]
    
    print(f"[INFO] Usando cámara {cam1_index} para entrada y cámara {cam2_index} para tracking")
    
    # Crear threads para cada cámara
    thread1 = threading.Thread(target=camera_1_entry, args=(cam1_index,))
    thread2 = threading.Thread(target=camera_2_tracking, args=(cam2_index,))
    
    # Iniciar threads
    thread1.daemon = True
    thread2.daemon = True
    thread1.start()
    thread2.start()
    
    try:
        # Mantener el programa corriendo
        while thread1.is_alive() and thread2.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[INFO] Deteniendo sistema...")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()