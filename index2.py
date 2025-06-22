import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import csv

# --- Cargar modelo YOLOv8 ---
model = YOLO("yolov8n.pt")  # Puedes usar yolov8s.pt para más precisión

# --- Inicializar DeepSORT ---
tracker = DeepSort(max_age=30, n_init=3)

# --- Cargar video desde archivo ---
video_path = "peoplewalking.mp4"  # Reemplaza con tu ruta al video
cap = cv2.VideoCapture(video_path)
tracking_log = []

frame_id = 0  # contador de frames

# --- Historial de trayectorias ---
track_history = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        if cls == 0:
            bbox = [x1, y1, x2 - x1, y2 - y1]
            detections.append((bbox, conf, "person"))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        cx = int((l + r) / 2)
        cy = int((t + b) / 2)

        # Guardar historial
        if track_id not in track_history:
            track_history[track_id] = []
        track_history[track_id].append((cx, cy))
        tracking_log.append({
            "track_id": track_id,
            "frame": frame_id,
            "x": cx,
            "y": cy
        })

        frame_id += 1

        # Dibujar caja e ID
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Dibujar trayectoria
        points = track_history[track_id]
        for j in range(1, len(points)):
            if points[j - 1] is None or points[j] is None:
                continue
            cv2.line(frame, points[j - 1], points[j], (255, 0, 0), 2)

    cv2.imshow("Seguimiento con trayectoria", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    

cap.release()
with open("tracking_data.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["track_id", "frame", "x", "y"])
    writer.writeheader()
    writer.writerows(tracking_log)
cv2.destroyAllWindows()