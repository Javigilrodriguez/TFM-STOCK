# stock_management.py
from datetime import datetime
import pandas as pd
import numpy as np
from demand_predictor import DemandPredictor
from production_optimizer import ProductionOptimizer
from data_processor import DataProcessor
from report_generator import ReportGenerator

class StockManagementSystem:
    """Sistema principal de gestión de stock"""
    
    def __init__(self, data_path: str, plan_date: str, expected_results_path: str = None):
        """Inicializa el sistema"""
        try:
            self.plan_date = datetime.strptime(plan_date, '%Y-%m-%d')
            
            # Cargar datos
            self.df = pd.read_csv(data_path, sep=';', encoding='latin1')
            print(f"Datos cargados de {data_path}: {len(self.df)} productos")
            
            # Cargar resultados esperados si existen
            self.expected_results = None
            if expected_results_path:
                self.expected_results = pd.read_csv(expected_results_path, 
                                                  sep=';', encoding='latin1')
                print(f"Resultados esperados cargados de {expected_results_path}")
            
            # Validar datos
            if not DataProcessor.validate_data(self.df):
                raise ValueError("Los datos no cumplen con los requisitos mínimos")
            
            # Inicializar componentes
            self.demand_predictor = DemandPredictor()
            self.optimizer = ProductionOptimizer()
            
            # Preparar datos
            self.df = DataProcessor.clean_data(self.df)
            self.df = DataProcessor.calculate_tendencies(self.df)
            
            # Entrenar modelo
            self.model = self._train_demand_model()

        except Exception as e:
            print(f"Error al inicializar el sistema: {e}")
            raise

    def _train_demand_model(self):
        """Entrena el modelo de predicción"""
        try:
            # Preparar features y target
            features = self.demand_predictor.prepare_features(self.df)
            target = self.df['M_Vta -15']
            
            # Entrenar modelo
            print("\nEntrenando modelo de demanda...")
            self.demand_predictor.fit(features, target)
            
            # Evaluar modelo
            metrics = self.demand_predictor.evaluate(features, target)
            
            print("\nMétricas del modelo:")
            print(f"MAE: {metrics['mae']:.2f}")
            print(f"RMSE: {metrics['rmse']:.2f}")
            print(f"R2: {metrics['r2']:.2f}")
            
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

    def optimize_production(self, available_hours: float, maintenance_hours: float):
        """Optimiza la producción"""
        try:
            # Predecir demanda
            self.predict_demand()
            
            # Optimizar producción
            production_hours, production_quantities, hours_used = self.optimizer.optimize(
                self.df, available_hours, maintenance_hours
            )
            
            # Crear plan de producción
            production_plan = self.df.copy()
            production_plan['CAJAS_PRODUCIR'] = production_quantities
            production_plan['HORAS_PRODUCCION'] = production_hours
            
            return production_plan

        except Exception as e:
            print(f"Error en optimización: {e}")
            raise

    def generate_production_report(self, production_plan: pd.DataFrame, 
                               output_path: str = "plan_produccion.csv"):
        """Genera reporte de producción"""
        try:
            # Generar reporte
            report = ReportGenerator.generate_report(production_plan, output_path)
            
            # Imprimir resumen
            ReportGenerator.print_report_summary(report)
            
            # Generar y mostrar alertas
            alerts = ReportGenerator.generate_alerts(production_plan)
            if alerts:
                print("\nALERTAS DETECTADAS:")
                for alert in alerts:
                    print(f"\n{alert['tipo']}:")
                    print(f"  {alert['mensaje']}")
            
            return report

        except Exception as e:
            print(f"Error generando informe: {e}")
            raise