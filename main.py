import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class StockManagementSystem:
    def __init__(self, data_path: str, plan_date: str, expected_results_path: str = None):
        """
        Inicializar el sistema de gestión de inventario
        
        Args:
            data_path: Ruta al archivo CSV de datos de prueba
            plan_date: Fecha de inicio de planificación (YYYY-MM-DD)
            expected_results_path: Ruta al archivo CSV con resultados esperados
        """
        self.SAFETY_STOCK_DAYS = 3
        self.MIN_PRODUCTION_HOURS = 2
        
        try:
            self.plan_date = datetime.strptime(plan_date, '%Y-%m-%d')
            # Cargar datos sin skiprows ya que los datos de prueba están limpios
            self.df = pd.read_csv(data_path, sep=';', encoding='latin1')
            
            # Cargar resultados esperados si existen
            self.expected_results = None
            if expected_results_path:
                self.expected_results = pd.read_csv(expected_results_path, sep=';', encoding='latin1')
            
            self._clean_data()
            self.model = self._train_demand_model()

        except Exception as e:
            print(f"Error al inicializar el sistema: {e}")
            raise

    def _clean_data(self):
        """Limpiar y preparar los datos para análisis"""
        try:
            # Eliminar filas completamente vacías
            self.df.dropna(how='all', inplace=True)
            
            # Convertir columnas numéricas
            num_columns = ['Cj/H', 'Disponible', 'Calidad', 'Stock Externo', 
                         'M_Vta -15', 'M_Vta -15 AA']
            
            for col in num_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col].replace({',': '.'}, regex=True), 
                                               errors='coerce')
            
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
            
            # Filtrar por fecha de OF si existe
            if '1ª OF' in self.df.columns:
                self.df['1ª OF'] = pd.to_datetime(self.df['1ª OF'], 
                                                errors='coerce', 
                                                format='%d/%m/%Y')
                self.df = self.df[(self.df['1ª OF'].isna()) | 
                                (self.df['1ª OF'] >= self.plan_date)]

        except Exception as e:
            print(f"Error en limpieza de datos: {e}")
            raise

    def _train_demand_model(self):
        """Entrenar modelo de predicción de demanda"""
        try:
            X = self.df[['M_Vta -15 AA']].values.reshape(-1, 1)
            y = self.df['M_Vta -15'].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"Modelo de demanda entrenado:")
            print(f"MSE: {mse:.2f}")
            print(f"R2: {r2:.2f}")

            return model

        except Exception as e:
            print(f"Error en entrenamiento del modelo: {e}")
            raise

    def predict_demand(self):
        """Predecir demanda con el modelo entrenado"""
        try:
            predictions = self.model.predict(
                self.df[['M_Vta -15 AA']].values.reshape(-1, 1)
            )
            self.df['PREDICTED_DEMAND'] = np.maximum(0, predictions)  # No permitir demanda negativa
            
            # Calcular error contra demanda real
            mae = np.mean(np.abs(self.df['PREDICTED_DEMAND'] - self.df['M_Vta -15']))
            print(f"Error medio absoluto en predicción: {mae:.2f}")
            
        except Exception as e:
            print(f"Error en predicción de demanda: {e}")
            raise

    def optimize_production(self, 
                          available_hours: float, 
                          maintenance_hours: float, 
                          pending_orders: dict = None):
        """
        Optimizar la producción usando programación lineal
        """
        try:
            self.predict_demand()
            net_hours = available_hours - maintenance_hours

            if net_hours <= 0:
                raise ValueError("No hay horas disponibles para producción")

            # Parámetros para optimización
            production_rates = self.df['Cj/H'].values
            current_stock = self.df['STOCK_TOTAL'].values
            demand = self.df['PREDICTED_DEMAND'].values

            # Limpiar valores no válidos
            demand = np.nan_to_num(demand, nan=0.0, posinf=0.0, neginf=0.0)
            current_stock = np.nan_to_num(current_stock, nan=0.0)
            
            # Función objetivo: minimizar déficit de inventario
            c = -1 * demand  # Maximizar producción basada en demanda

            # Restricciones
            A_ub = []
            b_ub = []

            # 1. Restricción de horas disponibles
            A_ub.append(1 / production_rates)
            b_ub.append(net_hours)

            # 2. Restricción de stock de seguridad
            safety_stock = demand * self.SAFETY_STOCK_DAYS
            for i in range(len(production_rates)):
                row = np.zeros(len(production_rates))
                row[i] = -1
                A_ub.append(row)
                b_ub.append(-(safety_stock[i] - current_stock[i]))

            # 3. Ajustar por pedidos pendientes
            if pending_orders:
                for code, quantity in pending_orders.items():
                    idx = self.df.index[self.df['COD_ART'] == code]
                    if len(idx) > 0:
                        current_stock[idx[0]] -= quantity

            # Resolver optimización
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, 
                           bounds=(0, None), method='highs')

            if not result.success:
                raise ValueError(f"Optimización no exitosa: {result.message}")

            # Crear plan de producción
            production_plan = self.df.copy()
            production_plan['CAJAS_PRODUCIR'] = result.x
            production_plan['HORAS_PRODUCCION'] = result.x / production_plan['Cj/H']

            # Validar contra resultados esperados si existen
            if self.expected_results is not None:
                self._validate_results(production_plan)

            return production_plan

        except Exception as e:
            print(f"Error en optimización: {e}")
            raise

    def _validate_results(self, production_plan: pd.DataFrame):
        """
        Validar resultados contra valores esperados
        """
        try:
            merged = production_plan.merge(
                self.expected_results,
                on='COD_ART',
                suffixes=('_pred', '_exp')
            )
            
            # Calcular métricas de error
            cajas_mae = mean_absolute_error(
                merged['CAJAS_PRODUCIR'], 
                merged['CAJAS_PRODUCIR_exp']
            )
            horas_mae = mean_absolute_error(
                merged['HORAS_PRODUCCION'], 
                merged['HORAS_ASIGNADAS']
            )
            
            print("\nValidación contra resultados esperados:")
            print(f"MAE Cajas a Producir: {cajas_mae:.2f}")
            print(f"MAE Horas de Producción: {horas_mae:.2f}")
            
            # Identificar diferencias significativas
            diff_threshold = 100  # Umbral para diferencias significativas
            significant_diffs = merged[
                abs(merged['CAJAS_PRODUCIR'] - merged['CAJAS_PRODUCIR_exp']) > diff_threshold
            ]
            
            if not significant_diffs.empty:
                print("\nDiferencias significativas encontradas:")
                for _, row in significant_diffs.iterrows():
                    print(f"Producto {row['COD_ART']}:")
                    print(f"  Predicho: {row['CAJAS_PRODUCIR']:.0f}")
                    print(f"  Esperado: {row['CAJAS_PRODUCIR_exp']:.0f}")
            
        except Exception as e:
            print(f"Error en validación: {e}")

    def generate_production_report(self, 
                                 production_plan: pd.DataFrame,
                                 output_path: str = "plan_produccion.csv"):
        """
        Generar informe detallado de producción
        """
        try:
            # Calcular métricas globales
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

            # Guardar plan detallado
            production_plan.to_csv(output_path, index=False, sep=';', encoding='latin1')
            print(f"\nPlan de producción guardado en: {output_path}")

            return report

        except Exception as e:
            print(f"Error generando informe: {e}")
            raise

if __name__ == "__main__":
    try:
        # Inicializar sistema con datos de prueba
        system = StockManagementSystem(
            data_path="stock_data_test.csv",
            plan_date="2024-12-31",
            expected_results_path="expected_results.csv"
        )

        # Parámetros de producción
        available_hours = 100
        maintenance_hours = 5
        pending_orders = {"000001": 50, "000002": 30}

        # Optimizar producción
        production_plan = system.optimize_production(
            available_hours,
            maintenance_hours,
            pending_orders
        )

        # Generar reporte
        report = system.generate_production_report(
            production_plan,
            "plan_produccion_test.csv"
        )

        print("\nReporte de producción:")
        for key, value in report.items():
            if key != 'products_detail':
                print(f"{key}: {value}")

    except Exception as e:
        print(f"Error en ejecución: {e}")