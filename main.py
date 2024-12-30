import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime

class StockManagementSystem:
    def __init__(self, data_path: str, plan_date: str):
        """
        Inicializar el sistema de gestión de inventario con los datos de entrada

        :param data_path: Ruta al archivo CSV que contiene los datos de inventario
        :param plan_date: Fecha de inicio de la planificación (formato YYYY-MM-DD)
        """
        try:
            self.plan_date = datetime.strptime(plan_date, '%Y-%m-%d')
            self.df = pd.read_csv(data_path, sep=';', encoding='latin1', skiprows=3, header=0)
            self.df.columns = self.df.iloc[0]  # Usar la primera fila como encabezado
            self.df = self.df[1:]  # Eliminar la fila usada como encabezado
            self.df.columns = self.df.columns.str.strip()  # Limpiar espacios

            self._clean_data()
            self.model = self._train_demand_model()

        except Exception as e:
            print(f"Error al cargar el archivo CSV: {e}")
            raise

        # Parámetros del sistema
        self.SAFETY_STOCK_DAYS = 3
        self.MIN_PRODUCTION_HOURS = 2

    def _clean_data(self):
        """
        Limpiar y preparar los datos para su análisis
        """
        try:
            self.df.dropna(how='all', inplace=True)
            num_columns = ['Cj/H', 'Disponible', 'Calidad', 'Stock Externo', 'M_Vta -15', 'M_Vta -15 AA']

            # Verificar si la columna VTA60D existe
            if 'VTA60D' in self.df.columns:
                num_columns.append('VTA60D')

            for col in num_columns:
                self.df[col] = pd.to_numeric(self.df[col].replace({',': '.'}, regex=True), errors='coerce')

            self.df.fillna({
                'Cj/H': 1,
                'Disponible': 0,
                'Calidad': 0,
                'Stock Externo': 0,
                'M_Vta -15': 0,
                'M_Vta -15 AA': 1,
                'VTA60D': 1  # Asignar 1 si la columna no está presente
            }, inplace=True)

            # Calcular el stock total
            self.df['STOCK_TOTAL'] = self.df['Disponible'] + self.df['Calidad'] + self.df['Stock Externo']

            # Filtrar productos descatalogados si la columna VTA60D existe
            if 'VTA60D' in self.df.columns:
                self.df = self.df[self.df['VTA60D'] > 0]

            # Filtrar órdenes de fabricación anteriores a la fecha de planificación
            self.df['1ª OF'] = pd.to_datetime(self.df['1ª OF'], errors='coerce', format='%d/%m/%Y')
            self.df = self.df[(self.df['1ª OF'].isna()) | (self.df['1ª OF'] >= self.plan_date)]
        except KeyError as e:
            print(f"Error en las columnas del DataFrame: {e}")
            raise

    def _train_demand_model(self):
        """
        Entrenar un modelo de predicción para la demanda usando datos históricos
        """
        try:
            X = self.df[['M_Vta -15 AA']].values.reshape(-1, 1)
            y = self.df['M_Vta -15'].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            print(f"Modelo de demanda entrenado con MSE: {mse:.2f}")

            return model
        except Exception as e:
            print(f"Error al entrenar el modelo de demanda: {e}")
            raise

    def predict_demand(self):
        """
        Predecir la demanda utilizando el modelo entrenado
        """
        try:
            self.df['PREDICTED_DEMAND'] = self.model.predict(self.df[['M_Vta -15 AA']].values.reshape(-1, 1))
        except Exception as e:
            print(f"Error al predecir la demanda: {e}")
            raise

    def add_new_product(self, code: str, name: str, estimated_sales: float, start_date: str):
        """
        Añadir un nuevo producto al sistema

        :param code: Código del artículo
        :param name: Nombre del artículo
        :param estimated_sales: Venta prevista
        :param start_date: Fecha de primera venta (formato YYYY-MM-DD)
        """
        try:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            if start_date >= self.plan_date:
                new_product = {
                    'COD_ART': code,
                    'NOM_ART': name,
                    'M_Vta -15': estimated_sales,
                    'M_Vta -15 AA': estimated_sales,
                    'Disponible': 0,
                    'Calidad': 0,
                    'Stock Externo': 0,
                    'Cj/H': 1,
                    'VTA60D': 1,
                    '1ª OF': None
                }
                new_product_df = pd.DataFrame([new_product], columns=self.df.columns)
                self.df = pd.concat([self.df, new_product_df], ignore_index=True)
            else:
                print("El producto no se puede añadir porque la fecha de primera venta es anterior a la planificación.")
        except Exception as e:
            print(f"Error al añadir un nuevo producto: {e}")
            raise

    def optimize_production(self, available_hours: float, maintenance_hours: float, pending_orders: dict = None):
        """
        Optimizar la producción usando programación lineal

        :param available_hours: Horas totales disponibles para producción
        :param maintenance_hours: Horas programadas para mantenimiento
        :param pending_orders: Pedidos pendientes por producto
        :return: Plan de producción optimizado
        """
        try:
            self.predict_demand()
            net_hours = available_hours - maintenance_hours

            production_rates = self.df['Cj/H'].values
            current_stock = self.df['STOCK_TOTAL'].values
            demand = self.df['PREDICTED_DEMAND'].values

            if net_hours <= 0:
                raise ValueError("No hay horas disponibles para producción.")

            # Validar y limpiar los valores de c, current_stock y demand
            demand = np.nan_to_num(demand, nan=0.0, posinf=0.0, neginf=0.0)
            current_stock = np.nan_to_num(current_stock, nan=0.0, posinf=0.0, neginf=0.0)
            c = -1 * np.minimum(demand, current_stock)

            A_ub = []
            b_ub = []

            A_ub.append(1 / production_rates)
            b_ub.append(net_hours)

            safety_stock = np.minimum(demand * self.SAFETY_STOCK_DAYS, current_stock)
            for i in range(len(production_rates)):
                row = np.zeros(len(production_rates))
                row[i] = -1
                A_ub.append(row)
                b_ub.append(-(safety_stock[i] - current_stock[i]))

            if pending_orders:
                for code, quantity in pending_orders.items():
                    idx = self.df.index[self.df['COD_ART'] == code]
                    if len(idx) > 0:
                        current_stock[idx[0]] -= quantity

            A_ub = np.array(A_ub, dtype=object)
            b_ub = np.array(b_ub)

            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')

            if not result.success:
                print("Error en la optimización:", result.message)
                raise ValueError("Optimización no exitosa")

            production_plan = self.df.copy()
            production_plan['CAJAS_PRODUCIR'] = result.x
            production_plan['HORAS_PRODUCCION'] = result.x / production_plan['Cj/H']

            return production_plan

        except Exception as e:
            print(f"Error al optimizar la producción: {e}")
            raise

    def generate_production_report(self, production_plan: pd.DataFrame):
        """
        Generar un informe completo de producción

        :param production_plan: Plan de producción optimizado
        :return: Resumen del informe de producción
        """
        try:
            report = {
                'total_production': production_plan['CAJAS_PRODUCIR'].sum(),
                'total_hours': production_plan['HORAS_PRODUCCION'].sum(),
                'products_to_produce': production_plan[production_plan['CAJAS_PRODUCIR'] > 0][['COD_ART', 'CAJAS_PRODUCIR']].to_dict('records')
            }

            production_plan.to_csv("plan_produccion.csv", index=False, sep=';', encoding='latin1')
            print("Informe de producción generado y guardado en 'plan_produccion.csv'.")

            return report

        except Exception as e:
            print(f"Error al generar el informe de producción: {e}")
            raise

if __name__ == "__main__":
    data_path = "stock_data.csv"
    plan_date = "2024-11-11"

    try:
        system = StockManagementSystem(data_path, plan_date)

        available_hours = 100
        maintenance_hours = 5
        pending_orders = {"244719": 50, "274756": 30}

        # Añadir un nuevo producto como ejemplo
        system.add_new_product("999999", "Producto Nuevo", 200, "2024-11-15")

        production_plan = system.optimize_production(available_hours, maintenance_hours, pending_orders)

        report = system.generate_production_report(production_plan)

        print(report)

    except Exception as e:
        print(f"Error durante la ejecución: {e}")