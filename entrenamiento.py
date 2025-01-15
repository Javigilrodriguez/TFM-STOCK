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
        
    def cargar_datasets(self):
        """Carga todos los datasets de la carpeta seleccionada"""
        carpeta = self.ruta_carpeta.get()
        if not carpeta:
            raise ValueError("Debe seleccionar una carpeta primero")
        
        datasets = []
        archivos_validos = [f for f in os.listdir(carpeta) 
                           if f.endswith(('.xlsx', '.xls', '.csv'))]
        
        for archivo in archivos_validos:
            ruta_completa = os.path.join(carpeta, archivo)
            try:
                # Leer el archivo CSV
                with open(ruta_completa, 'r', encoding='cp1252') as f:
                    lineas = f.readlines()
                
                # Encontrar la línea que contiene los encabezados
                indice_headers = -1
                for i, linea in enumerate(lineas):
                    if 'COD_ART;NOM_ART;COD_GRU' in linea:
                        indice_headers = i
                        break
                
                if indice_headers == -1:
                    print(f"No se encontraron encabezados en {archivo}")
                    continue
                
                # Crear un nuevo archivo temporal con solo los datos relevantes
                temp_file = os.path.join(os.path.dirname(ruta_completa), f'temp_{archivo}')
                with open(temp_file, 'w', encoding='cp1252') as f:
                    f.writelines(lineas[indice_headers:])
                
                # Leer el archivo temporal con pandas
                df = pd.read_csv(temp_file, sep=';', encoding='cp1252')
                
                # Eliminar el archivo temporal
                os.remove(temp_file)
                
                if not df.empty:
                    datasets.append(df)
                    print(f"Archivo cargado: {archivo}")
            
            except Exception as e:
                print(f"Error cargando {archivo}: {e}")
                continue
        
        if not datasets:
            raise ValueError("No se encontraron datos válidos en los archivos")
        
        # Combinar todos los datasets
        df_combinado = pd.concat(datasets, ignore_index=True)
        
        print(f"\nColumnas encontradas:")
        print(df_combinado.columns.tolist())
        print(f"Total de registros: {len(df_combinado)}")
        
        return df_combinado

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
        Prepara las características para el modelo considerando decimales con coma

        Args:
            df: DataFrame con los datos

        Returns:
            DataFrame con las características preparadas
        """
        try:
            def convertir_valor(valor):
                """Convierte valores con coma decimal a float"""
                if isinstance(valor, str):
                    # Reemplazar coma por punto, quitar espacios y miles
                    valor = valor.replace(',', '.').replace('.', '', valor.count('.')-1).replace(' ', '')
                try:
                    return float(valor)
                except (ValueError, TypeError):
                    return 0.0

            # Preparar características
            caracteristicas = pd.DataFrame()
            
            # Convertir columnas específicas, manejando separador decimal
            caracteristicas['venta_actual'] = df['M_Vta -15'].apply(convertir_valor)
            caracteristicas['venta_anterior'] = df['M_Vta -15 AA'].apply(convertir_valor)
            
            # Convertir columnas de stock, manejando valores en blanco o nulos
            caracteristicas['stock_total'] = (
                df['Disponible'].apply(convertir_valor) + 
                df['Calidad'].apply(convertir_valor) + 
                df['Stock Externo'].apply(convertir_valor)
            )
            
            # Características derivadas con manejo de conversión
            caracteristicas['dias_cobertura'] = np.where(
                caracteristicas['venta_actual'] > 0,
                caracteristicas['stock_total'] / caracteristicas['venta_actual'],
                0
            )
            
            caracteristicas['ratio_venta'] = np.where(
                caracteristicas['venta_anterior'] > 0,
                caracteristicas['venta_actual'] / caracteristicas['venta_anterior'],
                1
            )
            
            # Limpiar datos
            caracteristicas = caracteristicas.fillna(0)
            caracteristicas = caracteristicas.replace([np.inf, -np.inf], 0)
            
            # Guardar nombres de características
            self.features = caracteristicas.columns.tolist()
            
            return caracteristicas
            
        except Exception as e:
            print(f"Error preparando características de modelo: {e}")
            import traceback
            traceback.print_exc()
            raise

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
        Realiza predicciones usando el modelo

        Args:
            X: Características para predicción

        Returns:
            Array con predicciones
        """
        if not self.modelo_cargado:
            raise ValueError("No hay modelo válido cargado. Por favor, entrene o cargue un modelo primero.")
            
        try:
            # Función auxiliar para convertir valores con coma decimal
            def convertir_valor(valor):
                if isinstance(valor, str):
                    # Reemplazar coma por punto y quitar espacios
                    valor = valor.replace(',', '.').replace(' ', '')
                try:
                    return float(valor)
                except (ValueError, TypeError):
                    return 0.0

            # Aplicar conversión a todas las columnas
            X_converted = X.copy()
            for col in X.columns:
                X_converted[col] = X_converted[col].apply(convertir_valor)
            
            # Limpiar datos de entrada
            X_converted = X_converted.fillna(0)
            X_converted = X_converted.replace([np.inf, -np.inf], 0)
            
            predicciones = self.modelo.predict(X_converted)
            
            # Asegurar valores no negativos y redondear hacia arriba
            predicciones = np.ceil(np.maximum(predicciones, 0))
            
            return predicciones.astype(int)
            
        except Exception as e:
            print(f"Error en predicción: {e}")
            import traceback
            traceback.print_exc()
            raise