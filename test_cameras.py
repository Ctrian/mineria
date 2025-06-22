import cv2

def test_camera(index):
    """Prueba una cámara específica"""
    print(f"Probando cámara {index}...")
    cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        print(f"❌ No se pudo abrir la cámara {index}")
        return False
    
    # Obtener información de la cámara
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✅ Cámara {index} disponible:")
    print(f"   - Resolución: {width}x{height}")
    print(f"   - FPS: {fps}")
    
    # Mostrar video por 5 segundos
    print(f"   - Mostrando video de cámara {index} (presiona 'q' para siguiente cámara)")
    
    frame_count = 0
    while frame_count < 150:  # ~5 segundos a 30fps
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Error leyendo frame de cámara {index}")
            break
        
        # Agregar texto identificativo
        cv2.putText(frame, f"CAMARA {index}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Resolucion: {width}x{height}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow(f"Test Camara {index}", frame)
        
        if cv2.waitKey(33) & 0xFF == ord('q'):  # ~30fps
            break
        frame_count += 1
    
    cap.release()
    cv2.destroyWindow(f"Test Camara {index}")
    return True

def main():
    print("=== DETECTOR DE CÁMARAS ===")
    print("Este script detectará y probará todas las cámaras disponibles")
    print("Presiona 'q' para pasar a la siguiente cámara\n")
    
    available_cameras = []
    
    # Probar cámaras del 0 al 5
    for i in range(6):
        if test_camera(i):
            available_cameras.append(i)
        print()  # Línea en blanco
    
    print("=== RESUMEN ===")
    if len(available_cameras) >= 2:
        print(f"✅ Cámaras disponibles: {available_cameras}")
        print(f"✅ Puedes usar el sistema dual con:")
        print(f"   - Cámara {available_cameras[0]} para entrada (cam_1_entry)")
        print(f"   - Cámara {available_cameras[1]} para tracking (cam_2_tracking)")
        
        if len(available_cameras) > 2:
            print(f"   - Cámaras adicionales disponibles: {available_cameras[2:]}")
    else:
        print(f"❌ Solo se encontraron {len(available_cameras)} cámara(s): {available_cameras}")
        print("❌ Se necesitan al menos 2 cámaras para el sistema dual")
    
    print(f"\n💡 Tip: En Windows, la webcam integrada suele ser índice 0")
    print(f"💡 Tip: Las webcams USB externas suelen ser índices 1, 2, etc.")

if __name__ == "__main__":
    main()