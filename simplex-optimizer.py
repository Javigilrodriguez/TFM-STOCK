import pandas as pd
import numpy as np
from scipy.optimize import linprog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import os
import joblib
from datetime import datetime, timedelta

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductionOptimizer:
    def __init__(self):
        self.safety_stock_days = 3
        self.min_production_hours = 2
        self.available_hours_per_day = 24
        self.product_change_penalty = 0.15
        self.group_change_penalty = 0.30
        self.demand_model = None
        self.scaler = StandardScaler()
        self.model_path = 'models'
        
        # Crear directorio para modelos si no existe
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def prepare_training_data(self, df, lookback=15):
        """Prepara datos para entrenamiento del modelo de demanda"""
        try:
            features = []
            targets = []
            
            # Convertir columnas numéricas
            df['cajas_hora'] = pd.to_numeric(df['Cj/H'].str.replace(',', '.'), errors='coerce')
            df['ventas_15d'] = pd.to_numeric(df['M_Vta -15'].str.replace(',', '.'), errors='coerce')
            df['ventas_15d_anterior'] = pd.to_numeric(df['M_Vta -15 AA'].str.replace(',', '.'), errors='coerce')
            
            # Calcular tendencias y patrones
            df['tendencia'] = np.where(
                df['ventas_15d_anterior'] > 0,
                (df['ventas_15d'] - df['ventas_15d_anterior']) / df['ventas_15d_anterior'],
                0
            )
            
            # Crear features para cada producto
            for _, row in df.iterrows():
                if pd.notna(row['cajas_hora']) and pd.notna(row['ventas_15d']):
                    feature_vector = [
                        row['ventas_15d'],
                        row['ventas_15d_anterior'] if pd.notna(row['ventas_15d_anterior']) else 0,
                        row['cajas_hora'],
                        row['tendencia'],
                        row['ventas_15d'] / row['cajas_hora'] if row['cajas_hora'] > 0 else 0
                    ]
                    
                    target = row['ventas_15d']  # Usamos ventas actuales como target
                    
                    features.append(feature_vector)
                    targets.append(target)
            
            return np.array(features), np.array(targets)
            
        except Exception as e:
            logger.error(f"Error en preparación de datos de entrenamiento: {str(e)}")
            return None, None

    def train_demand_model(self, df):
        """Entrena el modelo de predicción de demanda"""
        try:
            logger.info("Preparando datos para entrenamiento...")
            X, y = self.prepare_training_data(df)
            if X is None or y is None or len(X) == 0:
                logger.error("No hay datos suficientes para entrenar")
                return False
                
            # Split datos
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Escalar features
            logger.info("Escalando features...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Entrenar modelo
            logger.info("Entrenando modelo Random Forest...")
            self.demand_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1  # Usar todos los cores disponibles
            )
            
            self.demand_model.fit(X_train_scaled, y_train)
            
            # Evaluar modelo
            y_pred = self.demand_model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"\nMétricas del modelo de demanda:")
            logger.info(f"MAE: {mae:.2f}")
            logger.info(f"RMSE: {rmse:.2f}")
            logger.info(f"R2: {r2:.2f}")
            
            # Guardar modelo
            self._save_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Error en entrenamiento del modelo: {str(e)}")
            return False

    def _save_model(self):
        """Guarda el modelo y el scaler"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f'demand_model_{timestamp}.joblib'
            scaler_filename = f'scaler_{timestamp}.joblib'
            
            model_path = os.path.join(self.model_path, model_filename)
            scaler_path = os.path.join(self.model_path, scaler_filename)
            
            joblib.dump(self.demand_model, model_path)
            joblib.dump(self.scaler, scaler_path)
            
            logger.info(f"Modelo guardado en: {model_path}")
            logger.info(f"Scaler guardado en: {scaler_path}")
            
        except Exception as e:
            logger.error(f"Error guardando modelo: {str(e)}")

    def predict_demand(self, product_data):
        """Predice la demanda futura usando el modelo entrenado"""
        try:
            if self.demand_model is None:
                logger.error("Modelo no entrenado")
                return None
                
            # Preparar features
            feature_vector = np.array([
                product_data['ventas_15d'],
                product_data['ventas_15d_anterior'] if pd.notna(product_data['ventas_15d_anterior']) else 0,
                product_data['cajas_hora'],
                product_data['tendencia'] if 'tendencia' in product_data else 0,
                product_data['ventas_15d'] / product_data['cajas_hora'] if product_data['cajas_hora'] > 0 else 0
            ]).reshape(1, -1)
            
            # Escalar y predecir
            feature_vector_scaled = self.scaler.transform(feature_vector)
            prediction = self.demand_model.predict(feature_vector_scaled)[0]
            
            return max(0, prediction)  # No permitir predicciones negativas
            
        except Exception as e:
            logger.error(f"Error en predicción de demanda: {str(e)}")
            return None

def load_dataset():
    """Carga el dataset desde la carpeta Dataset"""
    try:
        for file in os.listdir("Dataset"):
            if file.endswith('.csv'):
                file_path = os.path.join("Dataset", file)
                logger.info(f"Cargando archivo: {file}")
                return pd.read_csv(file_path, sep=';', encoding='latin1', skiprows=4)
        raise FileNotFoundError("No se encontró ningún archivo CSV en la carpeta Dataset")
    except Exception as e:
        logger.error(f"Error cargando dataset: {str(e)}")
        return None

def main():
    try:
        # Cargar datos
        df = load_dataset()
        if df is None:
            return
        
        # Crear optimizador
        optimizer = ProductionOptimizer()
        
        # Entrenar modelo
        logger.info("Iniciando entrenamiento del modelo...")
        if optimizer.train_demand_model(df):
            logger.info("Entrenamiento completado exitosamente")
            
            # Hacer algunas predicciones de prueba
            logger.info("\nProbando predicciones para algunos productos:")
            sample_products = df.head(3)  # Tomar 3 productos de ejemplo
            
            for _, product in sample_products.iterrows():
                prediction = optimizer.predict_demand(product)
                if prediction is not None:
                    logger.info(f"\nProducto: {product['NOM_ART']}")
                    logger.info(f"Ventas actuales: {product['M_Vta -15']}")
                    logger.info(f"Predicción: {prediction:.2f}")
        else:
            logger.error("Error en el entrenamiento del modelo")
            
    except Exception as e:
        logger.error(f"Error en ejecución: {str(e)}")

if __name__ == "__main__":
    main()