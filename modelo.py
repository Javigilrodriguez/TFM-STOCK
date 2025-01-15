from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import joblib
import json
import os
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple

class ModeloPrediccion:
    def __init__(self, ruta_modelo: str = 'modelos', modelo_especifico: Optional[str] = None):
        """
        Inicializa el modelo de predicción

        Args:
            ruta_modelo: Directorio donde se guardan los modelos
            modelo_especifico: Nombre específico del modelo a cargar (opcional)
        """
        self.ruta_modelo = ruta_modelo
        os.makedirs(ruta_modelo, exist_ok=True)
        
        # Inicializar modelo
        self.modelo = LinearRegression()
        self.metadata = {}
        self.modelo_cargado = False
        
        # Intentar cargar modelo existente
        if self.cargar_modelo(modelo_especifico):
            self.modelo_cargado = True
            print("Modelo existente cargado correctamente")
        else:
            print("No se encontró modelo existente o hubo error al cargar")
    
    def listar_modelos(self) -> List[Dict]:
        """Lista todos los modelos disponibles con sus metadatos"""
        modelos = []
        try:
            # Buscar archivos .joblib directamente
            archivos_modelo = [f for f in os.listdir(self.ruta_modelo) 
                             if f.endswith('_modelo.joblib')]
            
            for archivo in archivos_modelo:
                nombre_base = archivo.replace('_modelo.joblib', '')
                ruta_metadata = os.path.join(self.ruta_modelo, f"{nombre_base}_metadata.json")
                
                # Obtener fecha del archivo si no hay metadata
                fecha = datetime.fromtimestamp(
                    os.path.getmtime(os.path.join(self.ruta_modelo, archivo))
                ).strftime("%Y%m%d_%H%M%S")
                
                metadata = {}
                if os.path.exists(ruta_metadata):
                    try:
                        with open(ruta_metadata, 'r') as f:
                            metadata = json.load(f)
                    except:
                        metadata = {'fecha_entrenamiento': fecha}
                else:
                    metadata = {'fecha_entrenamiento': fecha}
                
                modelos.append({
                    'nombre': nombre_base,
                    'fecha': metadata.get('fecha_entrenamiento', fecha),
                    'metricas': metadata.get('metricas', {}),
                    'parametros': metadata.get('parametros', {})
                })
            
            print(f"Modelos encontrados: {len(modelos)}")
            return sorted(modelos, key=lambda x: x['fecha'], reverse=True)
            
        except Exception as e:
            print(f"Error listando modelos: {e}")
            return []
    
    def cargar_modelo(self, nombre_modelo: Optional[str] = None) -> bool:
        """
        Carga un modelo específico o el más reciente

        Args:
            nombre_modelo: Nombre específico del modelo a cargar

        Returns:
            bool: True si el modelo se cargó correctamente
        """
        try:
            modelos = self.listar_modelos()
            if not modelos:
                print("No se encontraron modelos guardados")
                return False
            
            # Seleccionar modelo
            if nombre_modelo:
                modelo_seleccionado = next(
                    (m for m in modelos if m['nombre'] == nombre_modelo), None)
                if not modelo_seleccionado:
                    print(f"Modelo {nombre_modelo} no encontrado")
                    return False
            else:
                modelo_seleccionado = modelos[0]  # El más reciente
            
            # Cargar el modelo
            ruta_modelo = os.path.join(
                self.ruta_modelo, 
                f"{modelo_seleccionado['nombre']}_modelo.joblib"
            )
            
            if not os.path.exists(ruta_modelo):
                print(f"Archivo de modelo no encontrado: {ruta_modelo}")
                return False
                
            self.modelo = joblib.load(ruta_modelo)
            self.metadata = modelo_seleccionado
            
            print(f"Modelo cargado: {modelo_seleccionado['nombre']}")
            print(f"Fecha: {modelo_seleccionado['fecha']}")
            
            return True
            
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            return False
    
    def guardar_modelo(self) -> bool:
        """
        Guarda el modelo entrenado y sus metadatos

        Returns:
            bool: True si el modelo se guardó correctamente
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ruta_base = os.path.join(self.ruta_modelo, f"modelo_{timestamp}")
            
            # Guardar el modelo
            joblib.dump(self.modelo, f"{ruta_base}_modelo.joblib")
            
            # Calcular R² score correctamente
            r2_score = None
            try:
                if hasattr(self.modelo, 'score_'):
                    r2_score = float(self.modelo.score_)
                elif hasattr(self.modelo, 'score'):
                    if not callable(self.modelo.score):
                        r2_score = float(self.modelo.score)
            except:
                pass
            
            # Preparar metadatos
            metadatos = {
                'fecha_entrenamiento': timestamp,
                'metricas': {
                    'r2_score': r2_score,
                    'coef': self.modelo.coef_.tolist() if hasattr(self.modelo, 'coef_') else None
                },
                'parametros': {
                    'dias_stock_seguridad': 3,
                    'horas_min_produccion': 2,
                    'features': getattr(self, 'features', None)
                }
            }
            
            # Guardar metadatos
            with open(f"{ruta_base}_metadata.json", 'w') as f:
                json.dump(metadatos, f, indent=2)
            
            print(f"Modelo guardado exitosamente: modelo_{timestamp}")
            return True
            
        except Exception as e:
            print(f"Error guardando modelo: {e}")
            traceback.print_exc()
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

            caracteristicas = pd.DataFrame()
            
            # Características base con conversión de valores
            caracteristicas['venta_actual'] = df['M_Vta -15'].apply(convertir_valor)
            caracteristicas['venta_anterior'] = df['M_Vta -15 AA'].apply(convertir_valor)
            caracteristicas['stock_total'] = (
                df['Disponible'].apply(convertir_valor) + 
                df['Calidad'].apply(convertir_valor) + 
                df['Stock Externo'].apply(convertir_valor)
            )
            
            # Características derivadas
            caracteristicas['dias_cobertura'] = caracteristicas['stock_total'] / np.where(
                caracteristicas['venta_actual'] > 0,
                caracteristicas['venta_actual'] / 15,
                1
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
            raise
    
    def entrenar(self, X: pd.DataFrame, y: pd.Series):
        """
        Entrena el modelo y lo guarda

        Args:
            X: Características de entrenamiento
            y: Variable objetivo
        """
        try:
            print("Iniciando entrenamiento del modelo...")
            
            # Asegurar que no hay valores infinitos o NaN
            X = X.fillna(0)
            y = y.fillna(0)
            
            self.modelo.fit(X, y)
            self.modelo_cargado = True
            
            # Calcular métricas básicas
            y_pred = self.modelo.predict(X)
            mse = np.mean((y - y_pred) ** 2)
            r2 = self.modelo.score(X, y)
            
            print(f"Entrenamiento completado:")
            print(f"MSE: {mse:.4f}")
            print(f"R² Score: {r2:.4f}")
            
            # Guardar el modelo
            if self.guardar_modelo():
                print("Modelo guardado exitosamente")
            else:
                print("Error al guardar el modelo")
            
        except Exception as e:
            print(f"Error en entrenamiento: {e}")
            raise
    
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
            # Limpiar datos de entrada
            X = X.fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
            
            predicciones = self.modelo.predict(X)
            
            # Asegurar valores no negativos y redondear hacia arriba
            predicciones = np.ceil(np.maximum(predicciones, 0))
            
            return predicciones.astype(int)
            
        except Exception as e:
            print(f"Error en predicción: {e}")
            raise
    
    def validar_predicciones(self, predicciones: np.ndarray, 
                           y_real: np.ndarray) -> Dict:
        """
        Valida la calidad de las predicciones

        Args:
            predicciones: Valores predichos
            y_real: Valores reales

        Returns:
            Diccionario con métricas de validación
        """
        try:
            error_abs = np.abs(predicciones - y_real)
            
            metricas = {
                'error_medio': float(np.mean(error_abs)),
                'error_max': float(np.max(error_abs)),
                'error_min': float(np.min(error_abs)),
                'predicciones_correctas': float(np.mean(error_abs == 0)) * 100
            }
            
            print("\nMétricas de validación:")
            for metrica, valor in metricas.items():
                print(f"{metrica}: {valor:.2f}")
            
            return metricas
            
        except Exception as e:
            print(f"Error en validación: {e}")
            return {}