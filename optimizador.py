import numpy as np
import pandas as pd
from scipy.optimize import linprog
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class Optimizador:
    def __init__(self, dias_stock_seguridad=3, horas_min_produccion=2):
        self.dias_stock_seguridad = dias_stock_seguridad
        self.horas_min_produccion = horas_min_produccion
        self.TOLERANCE = 1e-7
        self.debug = True
        self.historial_planificacion = []
        
    def optimizar_produccion(self, df: pd.DataFrame, horas_disponibles: float, 
                       horas_mantenimiento: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """Optimiza la producción usando método Simplex"""
        try:
            print("Columnas disponibles:", df.columns.tolist())
            print("Tipos de datos:\n", df.dtypes)
            
            df_proc = self._preparar_datos(df)
            
            print("\nDespués de preparar datos:")
            print("Columnas procesadas:", df_proc.columns.tolist())
            print("Tipos de datos procesados:\n", df_proc.dtypes)
            
            df_proc = self._preparar_datos(df)
            n_productos = len(df_proc)
            horas_netas = horas_disponibles - horas_mantenimiento
            
            # Función objetivo (minimizar déficit de stock)
            c = np.zeros(n_productos)
            for i in range(n_productos):
                demanda = df_proc['DEMANDA_PREDICHA'].iloc[i]
                stock_objetivo = demanda * self.dias_stock_seguridad
                stock_actual = df_proc['STOCK_TOTAL'].iloc[i]
                c[i] = -(stock_objetivo - stock_actual)
            
            # Restricciones
            A_ub = []  # Matriz de restricciones
            b_ub = []  # Vector de límites
            
            # Restricción de horas totales
            tasas_produccion = 1 / np.maximum(df_proc['Cj/H'].values, 1)
            A_ub.append(tasas_produccion)
            b_ub.append(horas_netas)
            
            # Restricción de producción mínima
            for i in range(n_productos):
                restriccion = np.zeros(n_productos)
                restriccion[i] = -1  # Producción debe ser mayor que mínimo
                A_ub.append(restriccion)
                b_ub.append(-df_proc['Cj/H'].iloc[i] * self.horas_min_produccion)
            
            # Optimizar usando Simplex
            resultado = linprog(
                c=c,
                A_ub=np.array(A_ub),
                b_ub=np.array(b_ub),
                bounds=[(0, None) for _ in range(n_productos)],
                method='revised simplex'
            )
            
            if not resultado.success:
                raise ValueError(f"No se encontró solución óptima: {resultado.message}")
            
            # Calcular resultados
            cantidades = resultado.x
            horas = cantidades * tasas_produccion
            
            # Guardar planificación para análisis
            self._guardar_planificacion(df_proc, horas, cantidades)
            
            return horas, cantidades, resultado.fun
            
        except Exception as e:
            if self.debug:
                print(f"Error en optimización: {str(e)}")
            raise
    
    def _preparar_datos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara los datos para optimización"""
        df_prep = df.copy()
        
        # Función de conversión robusta
        def convertir_valor(valor):
            if pd.isna(valor) or valor == '':
                return 0
            try:
                # Reemplazar coma por punto, quitar espacios
                if isinstance(valor, str):
                    valor = valor.replace(',', '.').replace(' ', '')
                return float(valor)
            except:
                return 0

        # Convertir valores numéricos
        columnas_num = ['Cj/H', 'Disponible', 'Calidad', 'Stock Externo', 'M_Vta -15']
        for col in columnas_num:
            if col in df_prep.columns:
                df_prep[col] = df_prep[col].apply(convertir_valor)
        
        # Calcular campos necesarios
        df_prep['STOCK_TOTAL'] = df_prep['Disponible'] + df_prep.get('Calidad', 0) + df_prep.get('Stock Externo', 0)
        
        if 'DEMANDA_PREDICHA' not in df_prep.columns:
            df_prep['DEMANDA_PREDICHA'] = df_prep.get('M_Vta -15', 0)
                
        return df_prep
        
    def _guardar_planificacion(self, df: pd.DataFrame, horas: np.ndarray, cantidades: np.ndarray):
        """Guarda resultados de planificación para análisis"""
        metricas = self._calcular_metricas(df, horas, cantidades)
        
        planificacion = {
            'fecha': datetime.now(),
            'df': df.copy(),
            'horas': horas.copy(),
            'cantidades': cantidades.copy(),
            'metricas': metricas
        }
        
        self.historial_planificacion.append(planificacion)
        
    def _calcular_metricas(self, df: pd.DataFrame, horas: np.ndarray, 
                          cantidades: np.ndarray) -> Dict:
        """Calcula métricas de la planificación"""
        stock_final = df['STOCK_TOTAL'] + cantidades
        
        # Días de cobertura
        cobertura = np.where(
            df['DEMANDA_PREDICHA'] > 0,
            stock_final / (df['DEMANDA_PREDICHA'] / 30),
            float('inf')
        )
        
        return {
            'cobertura_media': float(np.mean(cobertura[cobertura != float('inf')])),
            'productos_planificados': int(np.sum(cantidades > 0)),
            'horas_totales': float(np.sum(horas)),
            'cajas_totales': float(np.sum(cantidades))
        }
        
    def comparar_real_estimado(self, df_real: pd.DataFrame) -> Dict:
        """Compara resultados reales vs planificados"""
        if not self.historial_planificacion:
            raise ValueError("No hay planificación previa para comparar")
            
        ultima_plan = self.historial_planificacion[-1]
        df_plan = ultima_plan['df']
        
        # Alinear datos
        df_real = df_real.set_index('COD_ART')
        df_plan = df_plan.set_index('COD_ART')
        
        # Calcular diferencias
        diff_stock = df_real['STOCK_TOTAL'] - df_plan['STOCK_TOTAL']
        error_porc = (diff_stock / df_real['STOCK_TOTAL']) * 100
        
        return {
            'error_medio': float(diff_stock.mean()),
            'error_abs_medio': float(diff_stock.abs().mean()),
            'error_porcentual': float(error_porc.mean()),
            'productos_afectados': int(np.sum(diff_stock != 0))
        }
    
    def generar_grafico_evolucion(self) -> Optional[go.Figure]:
        """Genera gráfico de evolución de métricas"""
        if not self.historial_planificacion:
            return None
            
        fechas = [p['fecha'] for p in self.historial_planificacion]
        coberturas = [p['metricas']['cobertura_media'] for p in self.historial_planificacion]
        productos = [p['metricas']['productos_planificados'] for p in self.historial_planificacion]
        
        fig = go.Figure()
        
        # Cobertura media
        fig.add_trace(go.Scatter(
            x=fechas,
            y=coberturas,
            name='Cobertura Media (días)',
            mode='lines+markers'
        ))
        
        # Productos planificados
        fig.add_trace(go.Scatter(
            x=fechas,
            y=productos,
            name='Productos Planificados',
            mode='lines+markers',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='Evolución de la Planificación',
            yaxis=dict(title='Días de Cobertura'),
            yaxis2=dict(title='Productos', overlaying='y', side='right'),
            height=400,
            showlegend=True
        )
        
        return fig
        
    def generar_grafico_comparativo(self, df_real: Optional[pd.DataFrame] = None) -> go.Figure:
        """Genera gráfico comparativo de stock"""
        if not self.historial_planificacion:
            return None
            
        ultima_plan = self.historial_planificacion[-1]
        df_plan = ultima_plan['df']
        
        fig = go.Figure()
        
        # Stock planificado
        fig.add_trace(go.Bar(
            name='Stock Planificado',
            x=df_plan['COD_ART'],
            y=df_plan['STOCK_TOTAL'] + ultima_plan['cantidades'],
            marker_color='blue'
        ))
        
        # Stock real si está disponible
        if df_real is not None:
            fig.add_trace(go.Bar(
                name='Stock Real',
                x=df_real['COD_ART'],
                y=df_real['STOCK_TOTAL'],
                marker_color='red'
            ))
        
        fig.update_layout(
            title='Comparación de Stock',
            xaxis_title='Código Artículo',
            yaxis_title='Stock (unidades)',
            barmode='group',
            height=400,
            showlegend=True
        )
        
        return fig