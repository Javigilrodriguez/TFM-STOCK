import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DemandPredictor:
    """Clase para predicción de demanda con múltiples modelos"""
    
    def __init__(self):
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
            scores = cross_val_score(model, X_scaled, y, 
                                   cv=5, scoring='neg_mean_absolute_error')
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
        
        # Mostrar importancia de características
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_imp = pd.DataFrame({
                'feature': X.columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("\nImportancia de características:")
            for _, row in feature_imp.iterrows():
                print(f"{row['feature']}: {row['importance']:.3f}")
    
    def predict(self, X):
        """Realiza predicciones con el mejor modelo"""
        if self.best_model is None:
            raise ValueError("El modelo debe ser entrenado primero")
            
        X_scaled = self.scaler.transform(X)
        return np.maximum(0, self.best_model.predict(X_scaled))

class StockManagementSystem:
    """Sistema principal de gestión de stock"""
    
    def __init__(self, data_path: str, plan_date: str, expected_results_path: str = None):
        """Inicializa el sistema"""
        self.SAFETY_STOCK_DAYS = 3
        self.MIN_PRODUCTION_HOURS = 2
        
        try:
            # Cargar datos
            self.plan_date = datetime.strptime(plan_date, '%Y-%m-%d')
            self.df = pd.read_csv(data_path, sep=';', encoding='latin1')
            print(f"Datos cargados de {data_path}: {len(self.df)} productos")
            
            # Cargar resultados esperados si existen
            self.expected_results = None
            if expected_results_path:
                self.expected_results = pd.read_csv(expected_results_path, 
                                                  sep=';', encoding='latin1')
                print(f"Resultados esperados cargados de {expected_results_path}")
            
            # Inicializar predictor
            self.demand_predictor = DemandPredictor()
            
            # Preparar datos
            self._clean_data()
            self.model = self._train_demand_model()

        except Exception as e:
            print(f"Error al inicializar el sistema: {e}")
            raise

    def _clean_data(self):
        """Limpia y prepara los datos"""
        try:
            # Eliminar filas vacías
            self.df.dropna(how='all', inplace=True)
            
            # Convertir columnas numéricas
            num_columns = ['Cj/H', 'Disponible', 'Calidad', 'Stock Externo', 
                         'M_Vta -15', 'M_Vta -15 AA']
            
            for col in num_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col].replace({',': '.'}, regex=True), 
                                               errors='coerce')
                    print(f"Columna {col} convertida a numérica")
            
            # Rellenar valores nulos
            self.df.fillna({
                'Cj/H': 1,
                'Disponible': 0,
                'Calidad': 0,
                'Stock Externo': 0,
                'M_Vta -15': 0,
                'M_Vta -15 AA': 1
            }, inplace=True)
            
            # Calcular stock total
            self.df['STOCK_TOTAL'] = (self.df['Disponible'] + 
                                    self.df['Calidad'] + 
                                    self.df['Stock Externo'])
            
            print(f"Datos limpios: {len(self.df)} productos activos")

        except Exception as e:
            print(f"Error en limpieza de datos: {e}")
            raise

    def _train_demand_model(self):
        """Entrena el modelo de predicción"""
        try:
            features = self.demand_predictor.prepare_features(self.df)
            target = self.df['M_Vta -15']
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            
            # Entrenar modelo
            print("\nEntrenando modelo de demanda...")
            self.demand_predictor.fit(X_train, y_train)
            
            # Evaluar modelo
            y_pred = self.demand_predictor.predict(X_test)
            
            print("\nMétricas en conjunto de prueba:")
            print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
            print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
            print(f"R2: {r2_score(y_test, y_pred):.2f}")
            
            return self.demand_predictor
            
        except Exception as e:
            print(f"Error en entrenamiento del modelo: {e}")
            raise

    def predict_demand(self):
        """Predice la demanda futura"""
        try:
            features = self.demand_predictor.prepare_features(self.df)
            predictions = self.demand_predictor.predict(features)
            
            self.df['PREDICTED_DEMAND'] = predictions
            
            # Analizar predicciones
            mae = mean_absolute_error(self.df['M_Vta -15'], predictions)
            mape = np.mean(np.abs((self.df['M_Vta -15'] - predictions) / 
                                 self.df['M_Vta -15'])) * 100
            
            print("\nAnálisis de predicciones:")
            print(f"Error medio absoluto: {mae:.2f}")
            print(f"Error porcentual medio: {mape:.2f}%")
            
        except Exception as e:
            print(f"Error en predicción de demanda: {e}")
            raise

    def optimize_production(self, available_hours: float, maintenance_hours: float, 
                          pending_orders: dict = None):
        """Optimiza la producción"""
        try:
            self.predict_demand()
            net_hours = available_hours - maintenance_hours

            if net_hours <= 0:
                raise ValueError("No hay horas disponibles para producción")

            production_rates = np.maximum(self.df['Cj/H'].values, 1)
            current_stock = self.df['STOCK_TOTAL'].values
            demand = self.df['PREDICTED_DEMAND'].values

            # Limpiar valores
            demand = np.nan_to_num(demand, nan=0.0, posinf=0.0, neginf=0.0)
            current_stock = np.nan_to_num(current_stock, nan=0.0)
            
            print("\nDiagnóstico de optimización:")
            print(f"Horas netas disponibles: {net_hours}")
            print(f"Total productos: {len(production_rates)}")
            print(f"Demanda total: {demand.sum():.2f}")
            print(f"Stock total actual: {current_stock.sum():.2f}")

            # Priorizar productos
            priority_score = (demand * self.SAFETY_STOCK_DAYS - current_stock) / production_rates
            priority_order = np.argsort(-priority_score)
            
            # Asignar producción
            production_hours = np.zeros_like(demand)
            hours_used = 0
            
            for idx in priority_order:
                if hours_used >= net_hours:
                    break
                    
                target_stock = demand[idx] * self.SAFETY_STOCK_DAYS
                current = current_stock[idx]
                
                if current < target_stock:
                    hours_needed = (target_stock - current) / production_rates[idx]
                    
                    if hours_needed < self.MIN_PRODUCTION_HOURS:
                        if hours_used + self.MIN_PRODUCTION_HOURS <= net_hours:
                            hours_needed = self.MIN_PRODUCTION_HOURS
                        else:
                            continue
                    
                    hours_to_assign = min(hours_needed, net_hours - hours_used)
                    if hours_to_assign >= self.MIN_PRODUCTION_HOURS:
                        production_hours[idx] = hours_to_assign
                        hours_used += hours_to_assign

            # Calcular producción final
            production_quantities = production_hours * production_rates

            # Crear plan
            production_plan = self.df.copy()
            production_plan['CAJAS_PRODUCIR'] = production_quantities
            production_plan['HORAS_PRODUCCION'] = production_hours
            
            print("\nResumen de la solución:")
            print(f"Total cajas a producir: {production_quantities.sum():.2f}")
            print(f"Total horas asignadas: {hours_used:.2f}")
            print(f"Productos programados: {(production_quantities > 0).sum()}")

            return production_plan

        except Exception as e:
            print(f"Error en optimización: {e}")
            raise

    def generate_production_report(self, production_plan: pd.DataFrame, 
                                 output_path: str = "plan_produccion.csv"):
        """Genera reporte de producción"""
        try:
            # Calcular métricas
            total_production = production_plan['CAJAS_PRODUCIR'].sum()
            total_hours = production_plan['HORAS_PRODUCCION'].sum()
            products_to_produce = len(production_plan[production_plan['CAJAS_PRODUCIR'] > 0])
            
            report = {
                'total_production': total_production,
                'total_hours': total_hours,
                'products_to_produce': products_to_produce,
                'average_lot_size': total_production / products_to_produce if products_to_produce > 0 else 0,
                'products_detail': production_plan[
                    production_plan['CAJAS_PRODUCIR'] > 0
                ][['COD_ART', 'CAJAS_PRODUCIR', 'HORAS_PRODUCCION']].to_dict('records')
            }

            # Guardar plan
            production_plan.to_csv(output_path, index=False, sep=';', encoding='latin1')
            print(f"\nPlan de producción guardado en: {output_path}")

            return report

        except Exception as e:
            print(f"Error generando informe: {e}")
            raise

if __name__ == "__main__":
    try:
        print("Iniciando sistema de gestión de stock...")
        
        # Inicializar sistema
        system = StockManagementSystem(
            data_path="stock_data_test.csv",
            plan_date="2024-12-31",
            expected_results_path="expected_results.csv"
        )

        # Parámetros de producción
        available_hours = 100
        maintenance_hours = 5
        pending_orders = {"000001": 50, "000002": 30}

        print("\nOptimizando producción...")
        production_plan = system.optimize_production(
            available_hours,
            maintenance_hours,
            pending_orders
        )

        print("\nGenerando reporte...")
        report = system.generate_production_report(
            production_plan,
            "plan_produccion_test.csv"
        )

        print("\nReporte final de producción:")
        for key, value in report.items():
            if key != 'products_detail':
                print(f"{key}: {value}")
            else:
                print("\nDetalle de productos a producir:")
                for product in value:
                    print(f"  Código: {product['COD_ART']}")
                    print(f"  Cajas: {product['CAJAS_PRODUCIR']:.0f}")
                    print(f"  Horas: {product['HORAS_PRODUCCION']:.2f}")
                    print()

    except Exception as e:
        print(f"Error en ejecución: {e}")
        raise