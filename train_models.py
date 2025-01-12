# train_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime

from demand_predictor import DemandPredictor
from data_processor import DataProcessor

def load_training_data(path_training_data: str, path_historical_sales: str = None):
    """Carga y combina datos de entrenamiento"""
    try:
        # Cargar datos principales
        df_main = pd.read_csv(path_training_data, sep=';', encoding='latin1')
        print(f"Datos principales cargados: {len(df_main)} registros")
        
        # Cargar datos históricos si existen
        if path_historical_sales:
            df_hist = pd.read_csv(path_historical_sales, sep=';', encoding='latin1')
            print(f"Datos históricos cargados: {len(df_hist)} registros")
            
            # Aquí iría la lógica de combinación de datos históricos
            # Por ejemplo, agregar tendencias históricas, patrones estacionales, etc.
            
        return df_main
        
    except Exception as e:
        print(f"Error cargando datos: {e}")
        raise

def train_and_evaluate(training_data_path: str, historical_data_path: str = None, 
                      model_output_path: str = "trained_model.joblib"):
    """Entrena y evalúa los modelos de predicción"""
    try:
        print(f"Iniciando entrenamiento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Cargar datos
        df = load_training_data(training_data_path, historical_data_path)
        
        # Validar datos
        if not DataProcessor.validate_data(df):
            raise ValueError("Los datos no cumplen con los requisitos mínimos")
        
        # Limpiar y preparar datos
        df = DataProcessor.clean_data(df)
        df = DataProcessor.calculate_tendencies(df)
        
        # Inicializar predictor
        predictor = DemandPredictor()
        
        # Preparar features
        features = predictor.prepare_features(df)
        target = df['M_Vta -15']
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        print(f"\nDatos de entrenamiento: {X_train.shape[0]} muestras")
        print(f"Datos de prueba: {X_test.shape[0]} muestras")
        
        # Entrenar modelo
        print("\nEntrenando modelos...")
        predictor.fit(X_train, y_train)
        
        # Evaluar en conjunto de prueba
        print("\nEvaluando en conjunto de prueba...")
        y_pred = predictor.predict(X_test)
        
        # Calcular métricas
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print("\nResultados de evaluación:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2: {r2:.2f}")
        
        # Análisis detallado de errores
        errors = y_test - y_pred
        print("\nAnálisis de errores:")
        print(f"Error medio: {errors.mean():.2f}")
        print(f"Desviación estándar: {errors.std():.2f}")
        print(f"Error máximo: {errors.max():.2f}")
        print(f"Error mínimo: {errors.min():.2f}")
        
        # Guardar modelo
        print(f"\nGuardando modelo en {model_output_path}")
        joblib.dump(predictor, model_output_path)
        print("Modelo guardado exitosamente")
        
        return predictor, {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    except Exception as e:
        print(f"Error en entrenamiento: {e}")
        raise

def validate_predictions(predictor, validation_data_path: str):
    """Valida las predicciones en un conjunto de datos independiente"""
    try:
        # Cargar datos de validación
        df_val = pd.read_csv(validation_data_path, sep=';', encoding='latin1')
        print(f"\nDatos de validación cargados: {len(df_val)} registros")
        
        # Preparar features
        features = predictor.prepare_features(df_val)
        true_values = df_val['M_Vta -15']
        
        # Realizar predicciones
        predictions = predictor.predict(features)
        
        # Calcular métricas
        metrics = {
            'mae': mean_absolute_error(true_values, predictions),
            'rmse': np.sqrt(mean_squared_error(true_values, predictions)),
            'r2': r2_score(true_values, predictions)
        }
        
        print("\nResultados de validación:")
        print(f"MAE: {metrics['mae']:.2f}")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"R2: {metrics['r2']:.2f}")
        
        return metrics
        
    except Exception as e:
        print(f"Error en validación: {e}")
        raise

if __name__ == "__main__":
    try:
        # Configurar rutas de archivos
        TRAINING_DATA_PATH = "training_data.csv"
        HISTORICAL_DATA_PATH = "historical_sales.csv"  # Opcional
        VALIDATION_DATA_PATH = "validation_data.csv"   # Opcional
        MODEL_OUTPUT_PATH = "trained_model.joblib"
        
        # Entrenar y evaluar modelo
        predictor, training_metrics = train_and_evaluate(
            TRAINING_DATA_PATH,
            HISTORICAL_DATA_PATH,
            MODEL_OUTPUT_PATH
        )
        
        # Validar en conjunto independiente si existe
        if VALIDATION_DATA_PATH:
            validation_metrics = validate_predictions(predictor, VALIDATION_DATA_PATH)
        
        print("\nProceso de entrenamiento completado exitosamente")
        
    except Exception as e:
        print(f"Error en ejecución: {e}")
        raise