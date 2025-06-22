import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Cargar modelo YOLOv8
yolo_model = YOLO("yolov8n.pt")

# Inicializar Deep SORT con ReID basado en OSNet (ya incluido en deep_sort_realtime)
tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture("peoplewalking2.mp4")

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 and not ret2:
        break  # Salir si ambos videos terminaron

    if ret1:
        # Procesar frame1 igual que antes
        results1 = yolo_model(frame1)[0]

        detections1 = []
        for box in results1.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections1.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

        tracks1 = tracker.update_tracks(detections1, frame=frame1)

        for track in tracks1:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            cv2.rectangle(frame1, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame1, f"ID: {track_id}", (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Camara 1", frame1)

    if ret2:
        # Procesar frame2 igual que para frame1
        results2 = yolo_model(frame2)[0]

        detections2 = []
        for box in results2.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections2.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

        tracks2 = tracker.update_tracks(detections2, frame=frame2)

        for track in tracks2:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            cv2.rectangle(frame2, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame2, f"ID: {track_id}", (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Camara 2", frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
