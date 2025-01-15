from datetime import datetime, timedelta
import json
import os
from typing import List, Dict, Optional

class CalendarioProduccion:
    def __init__(self, ruta_archivo: str = 'calendario_config.json'):
        """
        Inicializa el calendario de producción
        
        Args:
            ruta_archivo: Ruta donde se guarda la configuración del calendario
        """
        self.ruta_archivo = ruta_archivo
        self.dias_habiles = {}  # {fecha: True/False}
        self.cargar_calendario()
    
    def cargar_calendario(self):
        """Carga la configuración guardada del calendario"""
        try:
            if os.path.exists(self.ruta_archivo):
                with open(self.ruta_archivo, 'r') as f:
                    self.dias_habiles = json.load(f)
            else:
                # Inicializar con días laborables por defecto (L-V)
                fecha_actual = datetime.now()
                for i in range(90):  # 3 meses hacia adelante
                    fecha = fecha_actual + timedelta(days=i)
                    # Por defecto, L-V son laborables
                    self.dias_habiles[fecha.strftime('%Y-%m-%d')] = fecha.weekday() < 5
                self.guardar_calendario()
                
        except Exception as e:
            print(f"Error cargando calendario: {e}")
            self.dias_habiles = {}
    
    def guardar_calendario(self):
        """Guarda la configuración del calendario"""
        try:
            with open(self.ruta_archivo, 'w') as f:
                json.dump(self.dias_habiles, f, indent=2)
        except Exception as e:
            print(f"Error guardando calendario: {e}")
    
    def es_dia_habil(self, fecha: datetime) -> bool:
        """
        Verifica si una fecha es hábil
        
        Args:
            fecha: Fecha a verificar
            
        Returns:
            bool: True si es día hábil, False si no
        """
        fecha_str = fecha.strftime('%Y-%m-%d')
        return self.dias_habiles.get(fecha_str, False)
    
    def establecer_dia_habil(self, fecha: datetime, es_habil: bool):
        """
        Establece si una fecha es hábil o no
        
        Args:
            fecha: Fecha a establecer
            es_habil: True si es hábil, False si no
        """
        fecha_str = fecha.strftime('%Y-%m-%d')
        self.dias_habiles[fecha_str] = es_habil
        self.guardar_calendario()
    
    def obtener_dias_habiles_rango(self, fecha_inicio: datetime, fecha_fin: datetime) -> List[datetime]:
        """
        Obtiene lista de días hábiles en un rango de fechas
        
        Args:
            fecha_inicio: Fecha de inicio del rango
            fecha_fin: Fecha fin del rango
            
        Returns:
            Lista de fechas hábiles en el rango
        """
        dias_habiles = []
        fecha_actual = fecha_inicio
        
        while fecha_actual <= fecha_fin:
            if self.es_dia_habil(fecha_actual):
                dias_habiles.append(fecha_actual)
            fecha_actual += timedelta(days=1)
        
        return dias_habiles
    
    def extender_calendario(self, dias: int = 90):
        """
        Extiende el calendario hacia adelante
        
        Args:
            dias: Número de días a extender
        """
        ultima_fecha = max(
            datetime.strptime(fecha, '%Y-%m-%d') 
            for fecha in self.dias_habiles.keys()
        ) if self.dias_habiles else datetime.now()
        
        for i in range(1, dias + 1):
            fecha = ultima_fecha + timedelta(days=i)
            fecha_str = fecha.strftime('%Y-%m-%d')
            if fecha_str not in self.dias_habiles:
                self.dias_habiles[fecha_str] = fecha.weekday() < 5
        
        self.guardar_calendario()