# grid_config.py
import json
import cv2
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

@dataclass
class ZoneConfig:
    id: str
    name: str
    x1: int
    y1: int
    x2: int
    y2: int
    color: Tuple[int, int, int] = (0, 255, 255)
    priority: int = 1  # Para alertas (1=baja, 2=media, 3=alta)
    max_capacity: int = 10  # Capacidad máxima de personas
    category: str = "general"  # Categoría del stand (comida, tecnología, etc.)

class GridConfigurator:
    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.zones: List[ZoneConfig] = []
        self.current_drawing = False
        self.start_point = None
        self.temp_zone = None
        
    def interactive_setup(self, sample_frame: np.ndarray):
        """Configuración interactiva del grid usando mouse"""
        print("\n=== CONFIGURACIÓN INTERACTIVA DEL GRID ===")
        print("Instrucciones:")
        print("1. Haz clic y arrastra para crear una zona")
        print("2. Presiona 'n' para nombrar la zona actual")
        print("3. Presiona 's' para guardar configuración")
        print("4. Presiona 'r' para resetear")
        print("5. Presiona 'q' para salir")
        
        cv2.namedWindow("Configuración de Grid", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Configuración de Grid", self._mouse_callback)
        
        while True:
            display_frame = sample_frame.copy()
            self._draw_zones(display_frame)
            
            # Mostrar zona temporal mientras se dibuja
            if self.temp_zone:
                cv2.rectangle(display_frame, 
                             (self.temp_zone.x1, self.temp_zone.y1),
                             (self.temp_zone.x2, self.temp_zone.y2),
                             self.temp_zone.color, 2)
                cv2.putText(display_frame, "Nueva zona", 
                           (self.temp_zone.x1, self.temp_zone.y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.temp_zone.color, 1)
            
            # Información en pantalla
            cv2.putText(display_frame, f"Zonas creadas: {len(self.zones)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, "Click y arrastra para crear zona", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Configuración de Grid", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n') and self.temp_zone:
                self._name_current_zone()
            elif key == ord('s'):
                self.save_config()
                print("Configuración guardada!")
            elif key == ord('r'):
                self.zones.clear()
                print("Grid reseteado!")
        
        cv2.destroyAllWindows()
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Callback para eventos del mouse"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE and self.current_drawing:
            if self.start_point:
                # Crear zona temporal
                x1, y1 = self.start_point
                x2, y2 = x, y
                
                # Asegurar que x1,y1 sea la esquina superior izquierda
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                
                zone_id = f"Z{len(self.zones) + 1}"
                color = self._get_next_color()
                
                self.temp_zone = ZoneConfig(
                    id=zone_id,
                    name=f"Zona {len(self.zones) + 1}",
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    color=color
                )
        
        elif event == cv2.EVENT_LBUTTONUP and self.current_drawing:
            if self.temp_zone and self._is_valid_zone(self.temp_zone):
                self.zones.append(self.temp_zone)
                print(f"Zona creada: {self.temp_zone.name}")
            
            self.current_drawing = False
            self.start_point = None
            self.temp_zone = None
    
    def _is_valid_zone(self, zone: ZoneConfig) -> bool:
        """Verifica si una zona es válida"""
        min_size = 50
        return (zone.x2 - zone.x1) > min_size and (zone.y2 - zone.y1) > min_size
    
    def _get_next_color(self) -> Tuple[int, int, int]:
        """Obtiene el siguiente color para una zona"""
        colors = [
            (255, 0, 0),    # Rojo
            (0, 255, 0),    # Verde
            (0, 0, 255),    # Azul
            (255, 255, 0),  # Amarillo
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cian
            (128, 0, 128),  # Púrpura
            (255, 165, 0),  # Naranja
            (0, 128, 128),  # Verde azulado
            (128, 128, 0),  # Oliva
        ]
        return colors[len(self.zones) % len(colors)]
    
    def _draw_zones(self, frame: np.ndarray):
        """Dibuja todas las zonas configuradas"""
        for zone in self.zones:
            cv2.rectangle(frame, (zone.x1, zone.y1), (zone.x2, zone.y2), zone.color, 2)
            cv2.putText(frame, zone.name, (zone.x1 + 5, zone.y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone.color, 1)
    
    def _name_current_zone(self):
        """Permite nombrar la zona actual"""
        if self.temp_zone:
            print(f"\nNombrando zona en posición ({self.temp_zone.x1}, {self.temp_zone.y1})")
            name = input("Nombre de la zona: ").strip()
            if name:
                self.temp_zone.name = name
            
            category = input("Categoría (comida/tecnología/ropa/general): ").strip()
            if category:
                self.temp_zone.category = category
            
            try:
                capacity = int(input("Capacidad máxima (default: 10): ") or "10")
                self.temp_zone.max_capacity = capacity
            except:
                pass
            
            try:
                priority = int(input("Prioridad 1-3 (default: 1): ") or "1")
                self.temp_zone.priority = max(1, min(3, priority))
            except:
                pass
    
    def load_predefined_configs(self):
        """Carga configuraciones predefinidas"""
        configs = {
            "3x3_grid": self._create_3x3_grid(),
            "4x4_grid": self._create_4x4_grid(),
            "exhibition_hall": self._create_exhibition_layout(),
            "store_layout": self._create_store_layout()
        }
        
        print("\nConfiguraciones predefinidas disponibles:")
        for i, (name, _) in enumerate(configs.items(), 1):
            print(f"{i}. {name}")
        
        try:
            choice = int(input("Selecciona una configuración (0 para personalizada): "))
            if 1 <= choice <= len(configs):
                config_name = list(configs.keys())[choice - 1]
                self.zones = configs[config_name]
                print(f"Configuración '{config_name}' cargada!")
                return True
        except:
            pass
        
        return False
    
    def _create_3x3_grid(self) -> List[ZoneConfig]:
        """Crea un grid 3x3 estándar"""
        zones = []
        rows, cols = 3, 3
        cell_width = self.frame_width // cols
        cell_height = self.frame_height // rows
        
        categories = ["tecnología", "comida", "ropa", "libros", "juguetes", 
                     "hogar", "deportes", "música", "arte"]
        
        for row in range(rows):
            for col in range(cols):
                zone_id = f"S{row}_{col}"
                name = f"Stand {row+1}-{col+1}"
                category = categories[(row * cols + col) % len(categories)]
                
                x1 = col * cell_width
                y1 = row * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                
                zones.append(ZoneConfig(
                    id=zone_id,
                    name=name,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    color=self._get_color_for_category(category),
                    category=category,
                    max_capacity=8
                ))
        
        return zones
    
    def _create_4x4_grid(self) -> List[ZoneConfig]:
        """Crea un grid 4x4 para espacios más grandes"""
        zones = []
        rows, cols = 4, 4
        cell_width = self.frame_width // cols
        cell_height = self.frame_height // rows
        
        for row in range(rows):
            for col in range(cols):
                zone_id = f"S{row}_{col}"
                name = f"Stand {row+1}-{col+1}"
                
                x1 = col * cell_width
                y1 = row * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                
                zones.append(ZoneConfig(
                    id=zone_id,
                    name=name,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    color=self._get_next_color(),
                    max_capacity=6
                ))
        
        return zones
    
    def _create_exhibition_layout(self) -> List[ZoneConfig]:
        """Crea un layout típico de exhibición"""
        zones = []
        
        # Entrada
        zones.append(ZoneConfig("entrada", "Entrada Principal", 
                               0, 0, self.frame_width//3, self.frame_height//4,
                               (0, 255, 0), category="entrada"))
        
        # Stands centrales
        center_w = self.frame_width // 2
        center_h = self.frame_height // 2
        
        stands = [
            ("stand_principal", "Stand Principal", center_w-100, center_h-100, center_w+100, center_h+100),
            ("info", "Información", 0, self.frame_height//2, self.frame_width//4, 3*self.frame_height//4),
            ("cafeteria", "Cafetería", 3*self.frame_width//4, 0, self.frame_width, self.frame_height//2),
            ("salida", "Salida", 2*self.frame_width//3, 3*self.frame_height//4, self.frame_width, self.frame_height)
        ]
        
        for zone_id, name, x1, y1, x2, y2 in stands:
            zones.append(ZoneConfig(zone_id, name, x1, y1, x2, y2, 
                                   self._get_next_color(), category="stand"))
        
        return zones
    
    def _create_store_layout(self) -> List[ZoneConfig]:
        """Crea un layout de tienda"""
        zones = []
        
        # Pasillos y secciones
        sections = [
            ("caja", "Área de Cajas", 0, 0, self.frame_width//4, self.frame_height//3),
            ("electronica", "Electrónicos", self.frame_width//4, 0, self.frame_width//2, self.frame_height//2),
            ("ropa", "Ropa", self.frame_width//2, 0, 3*self.frame_width//4, self.frame_height//2),
            ("comida", "Comida", 3*self.frame_width//4, 0, self.frame_width, self.frame_height//2),
            ("hogar", "Hogar", 0, self.frame_height//2, self.frame_width//2, self.frame_height),
            ("jardin", "Jardín", self.frame_width//2, self.frame_height//2, self.frame_width, self.frame_height)
        ]
        
        for zone_id, name, x1, y1, x2, y2 in sections:
            zones.append(ZoneConfig(zone_id, name, x1, y1, x2, y2,
                                   self._get_color_for_category(zone_id),
                                   category=zone_id, max_capacity=15))
        
        return zones
    
    def _get_color_for_category(self, category: str) -> Tuple[int, int, int]:
        """Obtiene color según categoría"""
        color_map = {
            "tecnología": (255, 0, 0),
            "comida": (0, 255, 0),
            "ropa": (255, 0, 255),
            "entrada": (0, 255, 255),
            "salida": (255, 255, 0),
            "electronica": (255, 100, 0),
            "hogar": (100, 255, 100),
            "jardin": (0, 200, 0)
        }
        return color_map.get(category, (128, 128, 128))
    
    def save_config(self, filename: str = "grid_config.json"):
        """Guarda la configuración a archivo JSON"""
        config_data = {
            "frame_dimensions": {
                "width": self.frame_width,
                "height": self.frame_height
            },
            "zones": [asdict(zone) for zone in self.zones],
            "created_at": json.dumps(str(np.datetime64('now')), default=str)
        }
        
        with open(filename, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"Configuración guardada en {filename}")
    
    def load_config(self, filename: str = "grid_config.json") -> bool:
        """Carga configuración desde archivo JSON"""
        try:
            with open(filename, 'r') as f:
                config_data = json.load(f)
            
            self.zones = []
            for zone_data in config_data.get("zones", []):
                zone = ZoneConfig(**zone_data)
                self.zones.append(zone)
            
            print(f"Configuración cargada desde {filename}")
            print(f"Zonas cargadas: {len(self.zones)}")
            return True
            
        except FileNotFoundError:
            print(f"Archivo {filename} no encontrado")
            return False
        except Exception as e:
            print(f"Error cargando configuración: {e}")
            return False

def main():
    """Configurador independiente"""
    print("=== CONFIGURADOR DE GRID ===")
    
    # Simular dimensiones de frame (puedes cambiar esto)
    frame_width = 120
    frame_height = 120
    
    configurator = GridConfigurator(frame_width, frame_height)
    
    # Intentar cargar configuración existente
    if configurator.load_config():
        print("¿Deseas usar la configuración existente? (s/n)")
        if input().lower() != 's':
            configurator.zones.clear()
    
    if not configurator.zones:
        print("\n1. Usar configuración predefinida")
        print("2. Configuración interactiva")
        choice = input("Selecciona opción (1-2): ")
        
        if choice == '1':
            configurator.load_predefined_configs()
        else:
            # Crear frame de muestra
            sample_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            sample_frame.fill(50)  # Fondo gris oscuro
            configurator.interactive_setup(sample_frame)
    
    # Mostrar resumen
    print(f"\n=== CONFIGURACIÓN FINAL ===")
    print(f"Total de zonas: {len(configurator.zones)}")
    for zone in configurator.zones:
        print(f"- {zone.name} ({zone.category}): {zone.x1},{zone.y1} -> {zone.x2},{zone.y2}")
    
    configurator.save_config()

if __name__ == "__main__":
    main()