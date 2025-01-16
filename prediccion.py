import numpy as np
from scipy.optimize import linprog

class Prediccion:
    """Clase para optimización de producción basada en predicciones"""
    
    def __init__(self, dias_stock_seguridad=3, horas_min_produccion=2):
        """
        Inicializa parámetros de optimización
        
        Args:
            dias_stock_seguridad: Días mínimos de stock a mantener
            horas_min_produccion: Horas mínimas de producción por lote
        """
        self.DIAS_STOCK_SEGURIDAD = dias_stock_seguridad
        self.HORAS_MIN_PRODUCCION = horas_min_produccion
        self.PENALIZACION_CAMBIO = 0.15  # 15% menos de horas disponibles por cambio
    
    def optimizar_produccion(self, df, horas_disponibles, horas_mantenimiento):
        """
        Optimiza la producción usando método Simplex
        
        Args:
            df: DataFrame con datos de stock y predicciones
            horas_disponibles: Horas totales disponibles
            horas_mantenimiento: Horas reservadas para mantenimiento
            
        Returns:
            tuple: (horas_producción, cantidades_producción, valor_objetivo)
        """
        try:
            n_productos = len(df)
            horas_netas = horas_disponibles - horas_mantenimiento
            
            if horas_netas <= 0:
                raise ValueError("No hay horas disponibles para producción")

            # 1. Preparar función objetivo (minimizar déficit de stock)
            c = np.zeros(n_productos)
            for i in range(n_productos):
                c[i] = -(df['DEMANDA_PREDICHA'].iloc[i] * self.DIAS_STOCK_SEGURIDAD - 
                        df['STOCK_TOTAL'].iloc[i])

            # 2. Restricciones
            restricciones = []
            valores_b = []

            # Restricción de horas disponibles
            tasas_produccion = 1 / np.maximum(df['Cj/H'].values, 1)
            restricciones.append(tasas_produccion)
            valores_b.append(horas_netas)

            # Restricción de producción mínima
            for i in range(n_productos):
                min_prod = np.zeros(n_productos)
                min_prod[i] = 1
                restricciones.append(min_prod)
                valores_b.append(df['Cj/H'].iloc[i] * self.HORAS_MIN_PRODUCCION)

            # 3. Optimización
            A = np.array(restricciones)
            b = np.array(valores_b)
            limites = [(0, None) for _ in range(n_productos)]
            
            resultado = linprog(
                c=c,
                A_ub=A,
                b_ub=b,
                bounds=limites,
                method='simplex'
            )

            if not resultado.success:
                raise ValueError(f"No se encontró solución óptima: {resultado.message}")

            # 4. Ajustar por cambios de producto
            cantidades = resultado.x
            horas = cantidades * tasas_produccion
            n_cambios = (cantidades > 0).sum() - 1
            
            if n_cambios > 0:
                # Reoptimizar con horas reducidas por cambios
                horas_penalizadas = horas_netas * (1 - self.PENALIZACION_CAMBIO * n_cambios)
                return self.optimizar_produccion(df, horas_penalizadas + horas_mantenimiento, 
                                              horas_mantenimiento)

            return horas, cantidades, resultado.fun

        except Exception as e:
            print(f"Error en optimización predd: {e}")
            raise
    
    def validar_solucion(self, df, cantidades_produccion):
        """
        Valida que la solución cumpla todas las restricciones
        
        Args:
            df: DataFrame con datos
            cantidades_produccion: Cantidades de producción calculadas
            
        Returns:
            tuple: (bool, list) - Válido y lista de mensajes
        """
        mensajes = []
        valido = True
        
        try:
            # 1. Validar stock de seguridad
            stock_final = df['STOCK_TOTAL'].values + cantidades_produccion
            demanda = df['DEMANDA_PREDICHA'].values
            violacion_stock = stock_final < (demanda * self.DIAS_STOCK_SEGURIDAD)
            
            if np.any(violacion_stock):
                valido = False
                productos = df[violacion_stock]['COD_ART'].tolist()
                mensajes.append(f"Stock insuficiente para: {productos}")

            # 2. Validar producción mínima
            violacion_min_prod = (
                (cantidades_produccion > 0) & 
                (cantidades_produccion < df['Cj/H'] * self.HORAS_MIN_PRODUCCION)
            )
            
            if np.any(violacion_min_prod):
                valido = False
                productos = df[violacion_min_prod]['COD_ART'].tolist()
                mensajes.append(f"Producción mínima no alcanzada para: {productos}")

            return valido, mensajes
            
        except Exception as e:
            return False, [f"Error en validación: {str(e)}"]