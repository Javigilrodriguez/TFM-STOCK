# demand_predictor.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class DemandPredictor:
    """Clase para predicción de demanda con múltiples modelos"""
    
    def __init__(self):
        """Inicializa los modelos de predicción"""
        self.models = {
            'linear': LinearRegression(),
            'huber': HuberRegressor(epsilon=1.35),
            'ransac': RANSACRegressor(random_state=42),
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.best_model = None
        self.scaler = StandardScaler()
        
    def prepare_features(self, df):
        """Prepara características avanzadas para el modelo"""
        features = pd.DataFrame()
        
        # Características básicas y derivadas
        features['venta_anterior'] = df['M_Vta -15 AA']
        features['ratio_cambio'] = df['M_Vta -15'] / df['M_Vta -15 AA']
        features['stock_total'] = df['Disponible'] + df['Calidad'] + df['Stock Externo']
        features['dias_cobertura'] = features['stock_total'] / (df['M_Vta -15'] / 15)
        features['tasa_produccion'] = df['Cj/H']
        
        # Limpiar datos
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(features.mean())
        
        return features
        
    def fit(self, X, y):
        """Entrena múltiples modelos y selecciona el mejor"""
        X_scaled = self.scaler.fit_transform(X)
        
        best_mae = float('inf')
        best_model_name = None
        
        print("\nEvaluación de modelos:")
        for name, model in self.models.items():
            scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
            mae = -scores.mean()
            
            print(f"{name.upper()}:")
            print(f"  MAE: {mae:.2f}")
            print(f"  Std: {scores.std():.2f}")
            
            if mae < best_mae:
                best_mae = mae
                best_model_name = name
        
        print(f"\nMejor modelo: {best_model_name.upper()} (MAE: {best_mae:.2f})")
        
        # Entrenar mejor modelo
        self.best_model = self.models[best_model_name]
        self.best_model.fit(X_scaled, y)
        
        # Mostrar importancia de características si está disponible
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_imp = pd.DataFrame({
                'feature': X.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("\nImportancia de características:")
            for _, row in feature_imp.iterrows():
                print(f"{row['feature']}: {row['importance']:.3f}")
                
        return self
    
    def predict(self, X):
        """Realiza predicciones con el mejor modelo"""
        if self.best_model is None:
            raise ValueError("El modelo debe ser entrenado primero")
            
        X_scaled = self.scaler.transform(X)
        return np.maximum(0, self.best_model.predict(X_scaled))

    def evaluate(self, X, y_true):
        """Evalúa el modelo con métricas detalladas"""
        if self.best_model is None:
            raise ValueError("El modelo debe ser entrenado primero")
            
        y_pred = self.predict(X)
        
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }