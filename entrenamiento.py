import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import joblib
import os
from typing import Dict, List, Tuple, Optional
import json

class ModeloEntrenamiento:
    """Clase para entrenamiento del modelo con persistencia y análisis temporal"""
    
    def __init__(self, ruta_modelo: str = 'modelos'):
        """
        Inicializa el modelo con soporte para persistencia
        
        Args:
            ruta_modelo: Directorio donde se guardarán los modelos entrenados
        """
        self.ruta_modelo = ruta_modelo
        self.modelo = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.patrones_temporales = {}
        self.ultimo_entrenamiento = None
        
        # Crear directorio si no existe
        os.makedirs(ruta_modelo, exist_ok=True)
        
        # Cargar modelo si existe
        self.cargar_modelo()
        
    def cargar_dataset_multiple(self, ruta_directorio: str) -> pd.DataFrame:
        """
        Carga múltiples archivos CSV/Excel de un directorio para entrenamiento
        
        Args:
            ruta_directorio: Ruta al directorio con archivos de datos
            
        Returns:
            DataFrame combinado con todos los datos
        """
        datos_combinados = []
        
        for archivo in os.listdir(ruta_directorio):
            if archivo.endswith(('.csv', '.xlsx', '.xls')):
                ruta_completa = os.path.join(ruta_directorio, archivo)
                try:
                    if archivo.endswith('.csv'):
                        df = pd.read_csv(ruta_completa, sep=';', encoding='latin1')
                    else:
                        df = pd.read_excel(ruta_completa)
                    
                    # Agregar metadatos temporales
                    df['fecha_archivo'] = os.path.getctime(ruta_completa)
                    datos_combinados.append(df)
                    
                except Exception as e:
                    print(f"Error cargando {archivo}: {e}")
                    continue
        
        if not datos_combinados:
            raise ValueError("No se encontraron archivos válidos en el directorio")
            
        return pd.concat(datos_combinados, ignore_index=True)

    def analizar_patrones_temporales(self, df: pd.DataFrame) -> Dict:
        """
        Analiza patrones temporales en los datos
        
        Args:
            df: DataFrame con datos históricos
            
        Returns:
            Diccionario con patrones identificados
        """
        patrones = {}
        
        # Convertir fechas a datetime si no lo están
        df['fecha'] = pd.to_datetime(df['fecha_archivo'])
        
        # Análisis por mes
        ventas_mensuales = df.groupby(df['fecha'].dt.month)['M_Vta -15'].mean()
        mes_max = ventas_mensuales.idxmax()
        mes_min = ventas_mensuales.idxmin()
        
        patrones['meses_alta_demanda'] = [
            mes for mes, ventas in ventas_mensuales.items()
            if ventas > ventas_mensuales.mean() + ventas_mensuales.std()
        ]
        
        patrones['meses_baja_demanda'] = [
            mes for mes, ventas in ventas_mensuales.items()
            if ventas < ventas_mensuales.mean() - ventas_mensuales.std()
        ]
        
        # Análisis por producto
        productos_estacionales = {}
        for producto in df['COD_ART'].unique():
            datos_producto = df[df['COD_ART'] == producto]
            if len(datos_producto) > 30:  # Mínimo de datos para análisis
                variacion = datos_producto.groupby(
                    datos_producto['fecha'].dt.month)['M_Vta -15'].std()
                if variacion.mean() > datos_producto['M_Vta -15'].mean() * 0.5:
                    productos_estacionales[producto] = True
        
        patrones['productos_estacionales'] = productos_estacionales
        
        return patrones

    def guardar_modelo(self):
        """Guarda el modelo entrenado y sus metadatos"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ruta_base = os.path.join(self.ruta_modelo, f"modelo_{timestamp}")
        
        # Guardar modelo y scaler
        joblib.dump(self.modelo, f"{ruta_base}_modelo.joblib")
        joblib.dump(self.scaler, f"{ruta_base}_scaler.joblib")
        
        # Guardar metadatos
        metadatos = {
            'fecha_entrenamiento': timestamp,
            'patrones_temporales': self.patrones_temporales,
            'metricas': self.modelo.score if hasattr(self.modelo, 'score') else None
        }
        
        with open(f"{ruta_base}_metadata.json", 'w') as f:
            json.dump(metadatos, f, indent=2)
            
    def cargar_modelo(self) -> bool:
        """
        Carga el último modelo guardado
        
        Returns:
            bool: True si se cargó un modelo, False si no
        """
        try:
            modelos = [f for f in os.listdir(self.ruta_modelo) 
                      if f.endswith('_modelo.joblib')]
            
            if not modelos:
                return False
                
            ultimo_modelo = max(modelos)
            base_nombre = ultimo_modelo.replace('_modelo.joblib', '')
            
            # Cargar modelo y scaler
            self.modelo = joblib.load(os.path.join(self.ruta_modelo, ultimo_modelo))
            self.scaler = joblib.load(os.path.join(
                self.ruta_modelo, f"{base_nombre}_scaler.joblib"))
            
            # Cargar metadatos
            with open(os.path.join(
                self.ruta_modelo, f"{base_nombre}_metadata.json"), 'r') as f:
                metadatos = json.load(f)
                
            self.patrones_temporales = metadatos['patrones_temporales']
            self.ultimo_entrenamiento = metadatos['fecha_entrenamiento']
            
            return True
            
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return False

    def necesita_reentrenamiento(self, datos_nuevos: pd.DataFrame) -> bool:
        """
        Determina si el modelo necesita reentrenamiento
        
        Args:
            datos_nuevos: DataFrame con nuevos datos
            
        Returns:
            bool: True si se recomienda reentrenar
        """
        if not self.ultimo_entrenamiento:
            return True
            
        # Verificar antigüedad del modelo
        dias_desde_entrenamiento = (datetime.now() - 
            datetime.strptime(self.ultimo_entrenamiento, "%Y%m%d_%H%M%S")).days
        if dias_desde_entrenamiento > 30:
            return True
            
        # Verificar cambios en patrones
        nuevos_patrones = self.analizar_patrones_temporales(datos_nuevos)
        if nuevos_patrones['meses_alta_demanda'] != self.patrones_temporales.get(
            'meses_alta_demanda', []):
            return True
            
        return False

    def preparar_caracteristicas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara características incluyendo información temporal
        
        Args:
            df: DataFrame con datos
            
        Returns:
            DataFrame con características preparadas
        """
        caracteristicas = pd.DataFrame()
        
        # Características base del modelo original
        caracteristicas['venta_anterior'] = df['M_Vta -15']
        caracteristicas['venta_anterior_anual'] = df['M_Vta -15 AA']
        caracteristicas['stock_total'] = (df['Disponible'] + 
                                        df['Calidad'] + 
                                        df['Stock Externo'])
        
        # Características derivadas
        caracteristicas['ratio_venta'] = df['M_Vta -15'] / df['M_Vta -15 AA']
        caracteristicas['dias_cobertura'] = caracteristicas['stock_total'] / (
            df['M_Vta -15'] / 15)
        
        # Características temporales mejoradas
        fecha_actual = datetime.now()
        caracteristicas['mes'] = fecha_actual.month
        caracteristicas['es_temporada_alta'] = caracteristicas['mes'].apply(
            lambda x: 1 if x in self.patrones_temporales.get('meses_alta_demanda', []) 
            else 0)
        
        # Características específicas por producto
        caracteristicas['es_producto_estacional'] = df['COD_ART'].apply(
            lambda x: 1 if x in self.patrones_temporales.get(
                'productos_estacionales', {}) else 0)
        
        # Limpiar datos
        caracteristicas = caracteristicas.replace([np.inf, -np.inf], np.nan)
        caracteristicas = caracteristicas.fillna(caracteristicas.mean())
        
        return caracteristicas

    def entrenar(self, X: pd.DataFrame, y: pd.Series):
        """
        Entrena el modelo y guarda los resultados
        
        Args:
            X: Características de entrenamiento
            y: Variable objetivo (ventas)
        """
        try:
            # Escalar características
            X_scaled = self.scaler.fit_transform(X)
            
            # Entrenar modelo
            print("Entrenando modelo...")
            self.modelo.fit(X_scaled, y)
            
            # Guardar modelo y metadatos
            self.guardar_modelo()
            
            # Mostrar importancia de características
            self._mostrar_importancia_caracteristicas(X)
            
        except Exception as e:
            print(f"Error en entrenamiento: {e}")
            raise

    def _mostrar_importancia_caracteristicas(self, X: pd.DataFrame):
        """Muestra la importancia de las características del modelo"""
        importancia = pd.DataFrame({
            'caracteristica': X.columns,
            'importancia': self.modelo.feature_importances_
        }).sort_values('importancia', ascending=False)
        
        print("\nImportancia de características:")
        for _, row in importancia.iterrows():
            print(f"{row['caracteristica']}: {row['importancia']:.3f}")

    def predecir(self, X: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones usando el modelo actual
        
        Args:
            X: Características para predicción
            
        Returns:
            Array con predicciones
        """
        try:
            X_scaled = self.scaler.transform(X)
            predicciones = self.modelo.predict(X_scaled)
            
            # Ajustar predicciones según patrones temporales
            mes_actual = datetime.now().month
            if mes_actual in self.patrones_temporales.get('meses_alta_demanda', []):
                predicciones *= 1.2  # Incrementar 20% en temporada alta
            elif mes_actual in self.patrones_temporales.get('meses_baja_demanda', []):
                predicciones *= 0.8  # Reducir 20% en temporada baja
                
            return np.maximum(0, predicciones)  # No permitir predicciones negativas
            
        except Exception as e:
            print(f"Error en predicción: {e}")
            raise