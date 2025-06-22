import cv2

url = "http://192.168.100.2:4747/video"  # cambia por la tuya

cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: No se pudo abrir el stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo leer frame")
        break

    cv2.imshow("Stream cámara celular", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
