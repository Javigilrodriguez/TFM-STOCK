import numpy as np
from scipy.optimize import linprog
import pandas as pd

class Optimizador:
    """Implementación del método Simplex para optimización"""
    
    def __init__(self, dias_stock_seguridad=3, horas_min_produccion=2):
        self.dias_stock_seguridad = dias_stock_seguridad
        self.horas_min_produccion = horas_min_produccion
    
    def optimizar_produccion(self, df, horas_disponibles, horas_mantenimiento):
        """
        Optimiza la producción usando método Simplex
        
        Args:
            df: DataFrame con datos y predicciones
            horas_disponibles: Horas totales disponibles
            horas_mantenimiento: Horas reservadas para mantenimiento
        
        Returns:
            tuple: (horas_producción, cantidades_producción, valor_objetivo)
        """
        # Preparar datos para Simplex
        n_productos = len(df)
        horas_netas = horas_disponibles - horas_mantenimiento
        
        # Función objetivo: minimizar déficit de stock
        c = np.zeros(n_productos)
        for i in range(n_productos):
            c[i] = -(df['DEMANDA_PREDICHA'].iloc[i] * self.dias_stock_seguridad - 
                    df['STOCK_TOTAL'].iloc[i])
        
        # Restricciones
        A_ub = []
        b_ub = []
        
        # Restricción de horas disponibles
        tasas_produccion = 1 / np.maximum(df['Cj/H'].values, 1)
        A_ub.append(tasas_produccion)
        b_ub.append(horas_netas)
        
        # Restricción de producción mínima
        for i in range(n_productos):
            restriccion = np.zeros(n_productos)
            restriccion[i] = 1
            A_ub.append(restriccion)
            b_ub.append(df['Cj/H'].iloc[i] * self.horas_min_produccion)
        
        # Optimización
        resultado = linprog(
            c=c,
            A_ub=np.array(A_ub),
            b_ub=np.array(b_ub),
            bounds=[(0, None) for _ in range(n_productos)],
            method='simplex'
        )
        
        if not resultado.success:
            raise ValueError(f"No se encontró solución óptima: {resultado.message}")
        
        cantidades = resultado.x
        horas = cantidades * tasas_produccion
        
        return horas, cantidades, resultado.fun