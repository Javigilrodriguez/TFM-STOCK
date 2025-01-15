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
            if isinstance(valor, (int, float)):
                return float(valor)
            elif isinstance(valor, str):
                # Limpiar el string
                valor_limpio = valor.replace(',', '.').strip()
                return float(valor_limpio)
            else:
                return 0.0
        except (ValueError, TypeError):
            return 0.0

    def _convertir_columna(self, serie):
        """Convierte una serie a valores numéricos"""
        if pd.api.types.is_numeric_dtype(serie):
            return serie.fillna(0)
        return serie.apply(self._convertir_a_numerico)

    def _preparar_datos(self, df):
        """Prepara y limpia los datos de entrada"""
        df_prep = df.copy()
        
        # Convertir columnas numéricas
        columnas_numericas = ['Cj/H', 'Disponible', 'Calidad', 'Stock Externo', 'M_Vta -15']
        for col in columnas_numericas:
            if col in df_prep.columns:
                df_prep[col] = self._convertir_columna(df_prep[col])
        
        # Calcular demanda predicha
        if 'DEMANDA_PREDICHA' not in df_prep.columns:
            df_prep['DEMANDA_PREDICHA'] = df_prep['M_Vta -15']
        
        self._log("Rangos de datos:")
        for col in columnas_numericas + ['DEMANDA_PREDICHA']:
            if col in df_prep.columns:
                self._log(f"{col}: [{df_prep[col].min():.2f}, {df_prep[col].max():.2f}]")
        
        return df_prep

    def _calcular_prioridad(self, df):
        """Calcula la prioridad de producción para cada producto"""
        df = df.copy()
        
        # Calcular días de cobertura actual
        df['dias_cobertura'] = np.where(
            df['DEMANDA_PREDICHA'] > 0,
            (df['Disponible'] + df['Calidad'] + df['Stock Externo']) / 
            (df['DEMANDA_PREDICHA'] / 30),
            float('inf')
        )
        
        # Calcular score de prioridad
        df['prioridad'] = np.where(
            df['DEMANDA_PREDICHA'] > 0,
            1 / (df['dias_cobertura'] + 1),
            0
        )
        
        return df

    def _dividir_en_lotes(self, df):
        """Divide los productos en lotes manejables basados en prioridad"""
        df = self._calcular_prioridad(df)
        
        # Ordenar por prioridad descendente
        df_sorted = df.sort_values('prioridad', ascending=False)
        
        # Filtrar productos relevantes
        df_relevant = df_sorted[
            (df_sorted['DEMANDA_PREDICHA'] > 0) & 
            (df_sorted['dias_cobertura'] < self.dias_stock_seguridad * 2)
        ]
        
        # Dividir en lotes
        lotes = []
        for i in range(0, len(df_relevant), self.MAX_BATCH_SIZE):
            lote = df_relevant.iloc[i:i + self.MAX_BATCH_SIZE]
            lotes.append(lote)
        
        self._log(f"Productos divididos en {len(lotes)} lotes")
        return lotes

    def _optimizar_lote(self, df_lote, horas_disponibles):
        """Optimiza un solo lote de productos"""
        try:
            n_productos = len(df_lote)
            
            # Preparar función objetivo
            c = np.zeros(n_productos)
            for i in range(n_productos):
                stock_objetivo = df_lote['DEMANDA_PREDICHA'].iloc[i] * self.dias_stock_seguridad
                stock_actual = (
                    df_lote['Disponible'].iloc[i] + 
                    df_lote['Calidad'].iloc[i] + 
                    df_lote['Stock Externo'].iloc[i]
                )
                c[i] = -(stock_objetivo - stock_actual)
            
            # Normalizar función objetivo
            c_max = np.abs(c).max()
            if c_max > 0:
                c = c / c_max
            
            # Preparar restricción de horas totales
            tasas_produccion = np.zeros(n_productos)
            for i in range(n_productos):
                if df_lote['Cj/H'].iloc[i] > self.TOLERANCE:
                    tasas_produccion[i] = 1 / df_lote['Cj/H'].iloc[i]
            
            A_ub = [tasas_produccion]
            b_ub = [horas_disponibles]
            
            # Optimizar
            resultado = linprog(
                c=c,
                A_ub=np.array(A_ub),
                b_ub=np.array(b_ub),
                bounds=[(0, None) for _ in range(n_productos)],
                method='interior-point',
                options={'maxiter': 5000, 'tol': 1e-4}
            )
            
            if not resultado.success:
                self._log(f"Optimización de lote falló: {resultado.message}")
                return None
            
            # Calcular cantidades y horas
            cantidades = resultado.x * c_max if c_max > 0 else resultado.x
            horas = cantidades * tasas_produccion
            
            # Limpiar valores pequeños
            mask = horas < self.TOLERANCE
            horas[mask] = 0
            cantidades[mask] = 0
            
            return horas, cantidades
            
        except Exception as e:
            self._log(f"Error optimizando lote: {str(e)}")
            return None

    def optimizar_produccion(self, df, horas_disponibles, horas_mantenimiento):
        """Optimiza la producción por lotes"""
        try:
            # Preparar datos
            df_prep = self._preparar_datos(df)
            
            # Dividir en lotes
            lotes = self._dividir_en_lotes(df_prep)
            
            if not lotes:
                self._log("No hay productos que requieran optimización")
                return np.zeros(len(df)), np.zeros(len(df)), 0
            
            # Optimizar cada lote
            horas_total = np.zeros(len(df))
            cantidades_total = np.zeros(len(df))
            horas_restantes = horas_disponibles - horas_mantenimiento
            
            for i, lote in enumerate(lotes):
                self._log(f"Optimizando lote {i+1}/{len(lotes)}")
                if horas_restantes <= 0:
                    break
                
                resultado = self._optimizar_lote(lote, horas_restantes)
                if resultado is not None:
                    horas_lote, cantidades_lote = resultado
                    
                    # Actualizar resultados
                    indices = lote.index
                    horas_total[indices] = horas_lote
                    cantidades_total[indices] = cantidades_lote
                    
                    # Actualizar horas restantes
                    horas_usadas = np.sum(horas_lote)
                    horas_restantes -= horas_usadas
                    self._log(f"Lote {i+1} completado. Horas usadas: {horas_usadas:.2f}")
            
            self._log(f"Optimización completada. Total horas asignadas: {np.sum(horas_total):.2f}")
            return horas_total, cantidades_total, 0
            
        except Exception as e:
            self._log(f"Error en optimización: {str(e)}")
            raise