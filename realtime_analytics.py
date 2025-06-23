# realtime_analytics.py
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time
import threading
from collections import defaultdict
import json

class RealTimeAnalytics:
    def __init__(self, db_path='tracking_data.db'):
        self.db_path = db_path
        self.running = True
        
    def get_live_stats(self):
        """Obtiene estadísticas en tiempo real"""
        conn = sqlite3.connect(self.db_path)
        
        # Personas activas (vistas en los últimos 5 minutos)
        active_query = """
            SELECT COUNT(DISTINCT person_id) as active_persons
            FROM tracking_events 
            WHERE timestamp > datetime('now', '-5 minutes')
        """
        
        # Zonas más visitadas
        zones_query = """
            SELECT zone_id, COUNT(*) as visits
            FROM tracking_events 
            WHERE event_type = 'zone_enter' 
            AND timestamp > datetime('now', '-1 hour')
            GROUP BY zone_id
            ORDER BY visits DESC
            LIMIT 5
        """
        
        # Tiempo promedio en zonas
        time_in_zones_query = """
            SELECT 
                zone_id,
                AVG(CAST((julianday(exit_time) - julianday(enter_time)) * 24 * 60 AS REAL)) as avg_minutes
            FROM (
                SELECT 
                    person_id,
                    zone_id,
                    timestamp as enter_time,
                    LEAD(timestamp) OVER (PARTITION BY person_id ORDER BY timestamp) as exit_time
                FROM tracking_events 
                WHERE event_type = 'zone_enter'
                AND timestamp > datetime('now', '-2 hours')
            ) 
            WHERE exit_time IS NOT NULL
            GROUP BY zone_id
        """
        
        active_persons = pd.read_sql_query(active_query, conn).iloc[0]['active_persons']
        top_zones = pd.read_sql_query(zones_query, conn)
        time_in_zones = pd.read_sql_query(time_in_zones_query, conn)
        
        conn.close()
        
        return {
            'active_persons': active_persons,
            'top_zones': top_zones.to_dict('records'),
            'avg_time_zones': time_in_zones.to_dict('records'),
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }
    
    def generate_heatmap_data(self):
        """Genera datos para mapa de calor de zonas"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT zone_id, COUNT(*) as intensity
            FROM tracking_events 
            WHERE event_type = 'zone_enter'
            AND timestamp > datetime('now', '-1 hour')
            GROUP BY zone_id
        """
        
        data = pd.read_sql_query(query, conn)
        conn.close()
        
        # Crear matriz 3x3 para el heatmap
        heatmap_matrix = [[0 for _ in range(3)] for _ in range(3)]
        
        for _, row in data.iterrows():
            if row['zone_id'] and row['zone_id'].startswith('S'):
                try:
                    parts = row['zone_id'].split('_')
                    if len(parts) == 2:
                        r, c = int(parts[0][1:]), int(parts[1])
                        if 0 <= r < 3 and 0 <= c < 3:
                            heatmap_matrix[r][c] = row['intensity']
                except:
                    continue
        
        return heatmap_matrix
    
    def get_person_journey(self, person_id):
        """Obtiene el recorrido de una persona específica"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT timestamp, event_type, zone_id, x_coordinate, y_coordinate
            FROM tracking_events 
            WHERE person_id = ?
            ORDER BY timestamp
        """
        
        journey = pd.read_sql_query(query, conn, params=(person_id,))
        conn.close()
        
        return journey.to_dict('records')
    
    def live_monitoring(self):
        """Monitor en tiempo real con actualización cada 10 segundos"""
        while self.running:
            try:
                stats = self.get_live_stats()
                heatmap = self.generate_heatmap_data()
                
                print(f"\n=== ESTADÍSTICAS EN TIEMPO REAL - {stats['timestamp']} ===")
                print(f"👥 Personas activas: {stats['active_persons']}")
                
                print("\n🔥 Top 5 Zonas más visitadas (última hora):")
                for zone in stats['top_zones']:
                    print(f"   {zone['zone_id']}: {zone['visits']} visitas")
                
                print("\n⏱️  Tiempo promedio en zonas:")
                for zone in stats['avg_time_zones']:
                    print(f"   {zone['zone_id']}: {zone['avg_minutes']:.1f} minutos")
                
                print("\n🗺️  Mapa de calor (matriz 3x3):")
                for i, row in enumerate(heatmap):
                    print(f"   Fila {i+1}: {row}")
                
                print("-" * 50)
                
                time.sleep(10)  # Actualizar cada 10 segundos
                
            except Exception as e:
                print(f"Error en monitoreo: {e}")
                time.sleep(5)
    
    def generate_report(self, hours_back=24):
        """Genera reporte detallado de las últimas N horas"""
        conn = sqlite3.connect(self.db_path)
        
        # Resumen general
        summary_query = f"""
            SELECT 
                COUNT(DISTINCT person_id) as total_persons,
                COUNT(*) as total_events,
                COUNT(CASE WHEN event_type = 'entry' THEN 1 END) as entries,
                COUNT(CASE WHEN event_type = 'zone_enter' THEN 1 END) as zone_visits
            FROM tracking_events 
            WHERE timestamp > datetime('now', '-{hours_back} hours')
        """
        
        # Análisis por hora
        hourly_query = f"""
            SELECT 
                strftime('%H', timestamp) as hour,
                COUNT(DISTINCT person_id) as unique_persons,
                COUNT(*) as events
            FROM tracking_events 
            WHERE timestamp > datetime('now', '-{hours_back} hours')
            GROUP BY strftime('%H', timestamp)
            ORDER BY hour
        """
        
        # Zonas más populares
        popular_zones_query = f"""
            SELECT 
                zone_id,
                COUNT(*) as visits,
                COUNT(DISTINCT person_id) as unique_visitors
            FROM tracking_events 
            WHERE event_type = 'zone_enter'
            AND timestamp > datetime('now', '-{hours_back} hours')
            GROUP BY zone_id
            ORDER BY visits DESC
        """
        
        summary = pd.read_sql_query(summary_query, conn)
        hourly = pd.read_sql_query(hourly_query, conn)
        popular_zones = pd.read_sql_query(popular_zones_query, conn)
        
        conn.close()
        
        # Generar reporte
        report = {
            'generated_at': datetime.now().isoformat(),
            'period_hours': hours_back,
            'summary': summary.to_dict('records')[0],
            'hourly_activity': hourly.to_dict('records'),
            'popular_zones': popular_zones.to_dict('records')
        }
        
        # Guardar como JSON
        report_filename = f"tracking_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Reporte guardado en: {report_filename}")
        return report
    
    def stop(self):
        """Detiene el monitoreo"""
        self.running = False

class WebDashboard:
    """Dashboard web simple usando Flask (opcional)"""
    def __init__(self, analytics: RealTimeAnalytics):
        self.analytics = analytics
        
    def create_api_endpoints(self):
        """Crea endpoints REST para consumir los datos"""
        try:
            from flask import Flask, jsonify
            from flask_cors import CORS
            
            app = Flask(__name__)
            CORS(app)
            
            @app.route('/api/stats')
            def get_stats():
                return jsonify(self.analytics.get_live_stats())
            
            @app.route('/api/heatmap')
            def get_heatmap():
                return jsonify({
                    'heatmap': self.analytics.generate_heatmap_data(),
                    'timestamp': datetime.now().isoformat()
                })
            
            @app.route('/api/person/<int:person_id>/journey')
            def get_person_journey(person_id):
                return jsonify({
                    'person_id': person_id,
                    'journey': self.analytics.get_person_journey(person_id),
                    'timestamp': datetime.now().isoformat()
                })
            
            @app.route('/api/report/<int:hours>')
            def get_report(hours):
                return jsonify(self.analytics.generate_report(hours))
            
            return app
            
        except ImportError:
            print("Flask no está instalado. Para usar el dashboard web, instala: pip install flask flask-cors")
            return None

def main():
    print("=== SISTEMA DE ANÁLISIS EN TIEMPO REAL ===")
    print("1. Monitor en consola")
    print("2. Generar reporte")
    print("3. Dashboard web (requiere Flask)")
    print("4. Análisis de persona específica")
    
    choice = input("Selecciona una opción (1-4): ")
    
    analytics = RealTimeAnalytics()
    
    if choice == '1':
        print("\n[INFO] Iniciando monitor en tiempo real...")
        print("Presiona Ctrl+C para detener")
        try:
            analytics.live_monitoring()
        except KeyboardInterrupt:
            print("\n[INFO] Monitor detenido")
    
    elif choice == '2':
        hours = int(input("¿Cuántas horas atrás analizar? (default: 24): ") or "24")
        report = analytics.generate_report(hours)
        print(f"\n=== REPORTE DE ÚLTIMAS {hours} HORAS ===")
        print(f"Total de personas: {report['summary']['total_persons']}")
        print(f"Total de eventos: {report['summary']['total_events']}")
        print(f"Entradas: {report['summary']['entries']}")
        print(f"Visitas a zonas: {report['summary']['zone_visits']}")
    
    elif choice == '3':
        dashboard = WebDashboard(analytics)
        app = dashboard.create_api_endpoints()
        if app:
            print("\n[INFO] Iniciando dashboard web en http://localhost:5000")
            print("Endpoints disponibles:")
            print("  - /api/stats - Estadísticas en tiempo real")
            print("  - /api/heatmap - Mapa de calor")
            print("  - /api/person/<id>/journey - Recorrido de persona")
            print("  - /api/report/<hours> - Reporte de N horas")
            app.run(debug=True, host='0.0.0.0', port=5000)
    
    elif choice == '4':
        person_id = int(input("ID de la persona a analizar: "))
        journey = analytics.get_person_journey(person_id)
        print(f"\n=== RECORRIDO DE PERSONA {person_id} ===")
        for event in journey:
            print(f"{event['timestamp']}: {event['event_type']} - {event['zone_id'] or 'N/A'}")

if __name__ == "__main__":
    main()