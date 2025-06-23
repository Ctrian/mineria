# main_system.py
"""
Sistema Principal de Tracking con Grid Inteligente
Integra todas las funcionalidades del sistema de tracking de personas
"""

import argparse
import sys
import os
from improved_dual_cameras import ImprovedTrackingSystem
from realtime_analytics import RealTimeAnalytics, WebDashboard
from grid_config import GridConfigurator
import cv2
import numpy as np

def setup_system():
    """Configuración inicial del sistema"""
    print("=== CONFIGURACIÓN DEL SISTEMA DE TRACKING ===")
    print()
    
    # Verificar dependencias
    try:
        import torch
        import ultralytics
        print("✅ PyTorch y YOLO disponibles")
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Instala las dependencias: pip install torch ultralytics")
        return False
    
    # Verificar cámaras
    available_cameras = []
    for i in range(4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    
    print(f"📹 Cámaras disponibles: {available_cameras}")
    
    if len(available_cameras) < 2:
        print("❌ Se necesitan al menos 2 cámaras para el sistema")
        return False
    
    # Configurar grid
    print("\n¿Deseas configurar el grid de zonas? (s/n)")
    if input().lower() == 'n':
        # Obtener dimensiones de la primera cámara
        cap = cv2.VideoCapture(available_cameras[0])
        ret, frame = cap.read()
        if ret:
            height, width = frame.shape[:2]
            configurator = GridConfigurator(width, height)
            
            if not configurator.load_config():
                print("Configurando grid...")
                configurator.load_predefined_configs()
                configurator.save_config()
        cap.release()
    
    return True

def run_tracking_system():
    """Ejecuta el sistema de tracking principal"""
    print("\n=== INICIANDO SISTEMA DE TRACKING ===")
    system = ImprovedTrackingSystem()
    system.run()

def run_analytics():
    """Ejecuta el sistema de análisis"""
    print("\n=== SISTEMA DE ANÁLISIS ===")
    analytics = RealTimeAnalytics()
    
    print("1. Monitor en tiempo real")
    print("2. Generar reporte")
    print("3. Dashboard web")
    print("4. Análisis de persona específica")
    
    choice = input("Selecciona opción (1-4): ")
    
    if choice == '1':
        try:
            analytics.live_monitoring()
        except KeyboardInterrupt:
            print("\nMonitor detenido")
    
    elif choice == '2':
        hours = int(input("Horas a analizar (default 24): ") or "24")
        analytics.generate_report(hours)
    
    elif choice == '3':
        dashboard = WebDashboard(analytics)
        app = dashboard.create_api_endpoints()
        if app:
            print("Dashboard disponible en http://localhost:5000")
            app.run(debug=True, host='0.0.0.0', port=5000)
    
    elif choice == '4':
        person_id = int(input("ID de persona: "))
        journey = analytics.get_person_journey(person_id)
        print(f"\nRecorrido de persona {person_id}:")
        for event in journey:
            print(f"  {event['timestamp']}: {event['event_type']} - {event['zone_id'] or 'N/A'}")

def run_configurator():
    """Ejecuta el configurador de grid"""
    print("\n=== CONFIGURADOR DE GRID ===")
    
    # Obtener dimensiones de cámara
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        height, width = frame.shape[:2]
        configurator = GridConfigurator(width, height)
        
        # Configuración interactiva
        configurator.interactive_setup(frame)
        cap.release()
    else:
        print("❌ No se pudo acceder a la cámara para configuración")

def show_system_status():
    """Muestra el estado del sistema"""
    print("\n=== ESTADO DEL SISTEMA ===")
    
    # Verificar archivos de configuración
    config_files = [
        ('grid_config.json', 'Configuración de grid'),
        ('person_db.pkl', 'Base de datos de personas'),
        ('tracking_data.db', 'Base de datos de tracking'),
        ('osnet_x1_0_imagenet.pth', 'Modelo ReID')
    ]
    
    for file, desc in config_files:
        status = "✅" if os.path.exists(file) else "❌"
        print(f"{status} {desc}: {file}")
    
    # Verificar cámaras
    available_cameras = []
    for i in range(4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    
    print(f"\n📹 Cámaras disponibles: {available_cameras}")
    
    # Verificar base de datos
    try:
        analytics = RealTimeAnalytics()
        stats = analytics.get_live_stats()
        print(f"👥 Personas activas: {stats['active_persons']}")
        print(f"📊 Zonas con actividad: {len(stats['top_zones'])}")
    except Exception as e:
        print(f"❌ Error accediendo a base de datos: {e}")

def main():
    parser = argparse.ArgumentParser(description='Sistema de Tracking con Grid Inteligente')
    parser.add_argument('--mode', choices=['tracking', 'analytics', 'config', 'status', 'setup'], 
                       default='tracking', help='Modo de operación')
    parser.add_argument('--auto-setup', action='store_true', help='Configuración automática')
    
    args = parser.parse_args()
    
    print("🎯 SISTEMA DE TRACKING CON GRID INTELIGENTE")
    print("=" * 50)
    
    if args.mode == 'setup' or args.auto_setup:
        if not setup_system():
            sys.exit(1)
        if args.auto_setup:
            args.mode = 'tracking'
    
    if args.mode == 'tracking':
        run_tracking_system()
    
    elif args.mode == 'analytics':
        run_analytics()
    
    elif args.mode == 'config':
        run_configurator()
    
    elif args.mode == 'status':
        show_system_status()
    
    else:
        # Menú interactivo
        while True:
            print("\n=== MENÚ PRINCIPAL ===")
            print("1. 🎯 Sistema de Tracking")
            print("2. 📊 Análisis de Datos")
            print("3. ⚙️  Configurar Grid")
            print("4. 📋 Estado del Sistema")
            print("5. 🔧 Configuración Inicial")
            print("6. ❌ Salir")
            
            choice = input("\nSelecciona una opción (1-6): ")
            
            if choice == '1':
                run_tracking_system()
            elif choice == '2':
                run_analytics()
            elif choice == '3':
                run_configurator()
            elif choice == '4':
                show_system_status()
            elif choice == '5':
                setup_system()
            elif choice == '6':
                print("¡Hasta luego!")
                break
            else:
                print("Opción no válida")

if __name__ == "__main__":
    main()