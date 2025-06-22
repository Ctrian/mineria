import cv2
import torch
from ultralytics import YOLO
from reid_utils import load_reid_model, get_embedding, load_database, is_same_person, get_dominant_color
import numpy as np
from datetime import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Usando dispositivo: {device}")

yolo = YOLO("yolov8n.pt")
reid_model = load_reid_model()
db = load_database()

cap = cv2.VideoCapture(0)  # Cambia por la fuente correcta

def color_distance(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))

# Para guardar las trayectorias: dict[id] = [(x,y), (x,y), ...]
trajectories = {}

# Parámetros
color_umbral = 50
embedding_umbral = 0.75
distancia_espacial_umbral = 100  # píxeles, ajustar según cámara

def distancia_puntos(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo(frame)[0]

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for box in results.boxes:
        if int(box.cls[0]) != 0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]

        embedding = get_embedding(reid_model, crop)
        color = get_dominant_color(crop)
        centro = ((x1 + x2) // 2, (y1 + y2) // 2)

        match_id = None
        min_score = float('inf')  # Para comparar y elegir mejor match
        for entry in db:
            if is_same_person(entry['embedding'], embedding, threshold=embedding_umbral):
                dist_c = color_distance(entry.get('dominant_color', [0, 0, 0]), color)
                if dist_c > color_umbral:
                    continue

                # Ver si la persona está cerca de su última posición (si existe)
                last_pos = trajectories.get(entry['id'], [])
                if last_pos:
                    dist_pos = distancia_puntos(last_pos[-1], centro)
                    if dist_pos > distancia_espacial_umbral:
                        continue
                else:
                    dist_pos = 0  # sin historial, lo tomamos

                score = dist_c + dist_pos  # una métrica simple a minimizar

                if score < min_score:
                    min_score = score
                    match_id = entry['id']

        if match_id is None:
            match_id = -1  # Persona desconocida

        # Guardar posición para dibujo
        if match_id != -1:
            if match_id not in trajectories:
                trajectories[match_id] = []
            trajectories[match_id].append(centro)

        color_rect = (0, 255, 0) if match_id != -1 else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_rect, 2)
        cv2.putText(frame, f"ID: {match_id}", (x1, y1 - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_rect, 2)
        cv2.putText(frame, current_time, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

   
    cv2.imshow("Cámara 2 - Tracking Mejorado", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
