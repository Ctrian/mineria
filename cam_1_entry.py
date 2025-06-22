import cv2
import torch
from ultralytics import YOLO
from reid_utils import load_reid_model, get_embedding, load_database, save_database, is_same_person, get_dominant_color

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Usando dispositivo: {device}")

# Cargar modelos
yolo = YOLO("yolov8n.pt")
reid_model = load_reid_model()
db = load_database()

# Captura de cámara
cap = cv2.VideoCapture(0)

current_max_id = max([p['id'] for p in db], default=0)
frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1
    results = yolo(frame)[0]

    for box in results.boxes:
        if int(box.cls[0]) != 0:  # solo personas
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]

        embedding = get_embedding(reid_model, crop)
        color = get_dominant_color(crop)  # EXTRA: color dominante

        match_id = None
        for entry in db:
            # Comparar embedding y color (puedes hacer más reglas aquí)
            if is_same_person(entry['embedding'], embedding, threshold=0.75):
                # Opcional: compara color también para confirmar
                # Puedes hacer algo como distancia euclidiana en color y definir un umbral
                match_id = entry['id']
                break

        if match_id is None:
            current_max_id += 1
            match_id = current_max_id
            db.append({'id': match_id, 'embedding': embedding, 'dominant_color': color})
            save_database(db)
            print(f"[NUEVO] Persona registrada con ID {match_id}, color dominante: {color}")
        else:
            print(f"[INFO] Persona ya registrada con ID {match_id}")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {match_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Cámara 1 - Entrada", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
