import numpy as np
from scipy.optimize import linprog, OptimizeWarning
import warnings
from typing import Tuple, Optional
import pandas as pd

class Optimizador:
    """Implementación mejorada del método Simplex con manejo robusto de datos"""
    
    def __init__(self, dias_stock_seguridad=3, horas_min_produccion=2):
        self.dias_stock_seguridad = dias_stock_seguridad
        self.horas_min_produccion = horas_min_produccion
        self.debug = True
        self.TOLERANCE = 1e-7
        self.MAX_BATCH_SIZE = 100
    
    def _log(self, mensaje):
        if self.debug:
            print(f"DEBUG: {mensaje}")

    def _convertir_a_numerico(self, valor):
        """Convierte un valor a numérico de forma robusta"""
        try:
            # Manejar valores nulos o vacíos
            if pd.isna(valor) or valor == '' or valor is None:
                return 0.0
            
            # Si ya es numérico, convertir directamente
            if isinstance(valor, (int, float)):
                return float(valor)
            
            # Convertir a cadena para procesar
            valor_str = str(valor).strip()
            
            # Reemplazar coma por punto decimal
            valor_str = valor_str.replace(',', '.')
            
            # Eliminar separadores de miles y otros caracteres no numéricos
            valor_str = ''.join(c for c in valor_str if c.isdigit() or c in '.-')
            
            return float(valor_str) if valor_str else 0.0
        except (ValueError, TypeError):
            return 0.0

    def _preparar_datos(self, df):
        """Prepara y limpia los datos de entrada"""
        df_prep = df.copy()
        
        # Columnas a convertir
        columnas_numericas = [
            'Cj/H', 'Disponible', 'Calidad', 'Stock Externo', 
            'M_Vta -15', 'M_Vta -15 AA', 'Vta -15', 'Vta -15 AA'
        ]
        
        # Convertir columnas
        for col in columnas_numericas:
            if col in df_prep.columns:
                # Aplicar conversión solo si la columna existe
                df_prep[col] = df_prep[col].apply(self._convertir_a_numerico)
            else:
                # Si la columna no existe, crearla con ceros
                df_prep[col] = 0
        
        # Calcular stock total
        df_prep['STOCK_TOTAL'] = (
            df_prep['Disponible'] + 
            df_prep['Calidad'] + 
            df_prep['Stock Externo']
        )
        
        # Establecer demanda predicha
        if 'DEMANDA_PREDICHA' not in df_prep.columns:
            # Intentar usar M_Vta -15 o su equivalente
            df_prep['DEMANDA_PREDICHA'] = df_prep.get('M_Vta -15', df_prep.get('Vta -15', 0))
        
        # Depuración de datos
        self._log("Columnas después de preparación:")
        for col in ['Disponible', 'STOCK_TOTAL', 'DEMANDA_PREDICHA']:
            self._log(f"{col}: min={df_prep[col].min()}, max={df_prep[col].max()}")
        
        return df_prep

    def optimizar_produccion(self, df, horas_disponibles, horas_mantenimiento):
        """Optimiza la producción usando método Simplex"""
        try:
            # Preparar datos
            df_prep = self._preparar_datos(df)
            
            n_productos = len(df_prep)
            horas_netas = horas_disponibles - horas_mantenimiento
            
            # Función objetivo
            c = np.zeros(n_productos)
            for i in range(n_productos):
                demanda = df_prep['DEMANDA_PREDICHA'].iloc[i]
                stock_objetivo = demanda * self.dias_stock_seguridad
                stock_actual = df_prep['STOCK_TOTAL'].iloc[i]
                c[i] = -(stock_objetivo - stock_actual)
            
            # Restricciones
            A_ub = []
            b_ub = []
            
            # Restricción de horas totales
            tasas_produccion = 1 / np.maximum(df_prep['Cj/H'].values, 0.01)
            A_ub.append(tasas_produccion)
            b_ub.append(horas_netas)
            
            # Restricciones de producción mínima
            for i in range(n_productos):
                restriccion = np.zeros(n_productos)
                restriccion[i] = -1
                A_ub.append(restriccion)
                b_ub.append(-df_prep['Cj/H'].iloc[i] * self.horas_min_produccion)
            
            # Optimizar
            resultado = linprog(
                c=c,
                A_ub=np.array(A_ub),
                b_ub=np.array(b_ub),
                bounds=[(0, None) for _ in range(n_productos)],
                method='revised simplex',
                options={'disp': False, 'maxiter': 5000}
            )
            
            if not resultado.success:
                self._log(f"Optimización falló: {resultado.message}")
                return (np.zeros(n_productos), 
                        np.zeros(n_productos), 
                        0)
            
            # Calcular resultados
            cantidades = resultado.x
            horas = cantidades * tasas_produccion
            
            return horas, cantidades, resultado.fun
            
        except Exception as e:
            self._log(f"Error en optimización opt: {str(e)}")
            raise