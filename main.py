import numpy as np
import pandas as pd
from scipy.optimize import linprog
from typing import Dict, Any

class StockManagementSystem:
    def __init__(self, data_path: str):
        """
        Inicializar el sistema de gestión de inventario con los datos de entrada

        :param data_path: Ruta al archivo CSV que contiene los datos de inventario
        """
        try:
            # Leer el archivo CSV
            self.df = pd.read_csv(data_path, sep=';', encoding='latin1', skiprows=3, header=0)
            self.df.columns = self.df.iloc[0]  # Usar la primera fila como encabezado
            self.df = self.df[1:]  # Eliminar la fila usada como encabezado
            self.df.columns = self.df.columns.str.strip()  # Limpiar espacios

            # Limpiar y preparar los datos
            self._clean_data()

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
            # Eliminar filas vacías
            self.df.dropna(how='all', inplace=True)

            # Convertir columnas relevantes a numéricas
            num_columns = ['Cj/H', 'Disponible', 'Calidad', 'Stock Externo', 'M_Vta -15', 'M_Vta -15 AA']
            for col in num_columns:
                self.df[col] = pd.to_numeric(self.df[col].replace({',': '.'}, regex=True), errors='coerce')

            # Reemplazar valores nulos
            self.df.fillna({
                'Cj/H': 1,  # Producción mínima
                'Disponible': 0,
                'Calidad': 0,
                'Stock Externo': 0,
                'M_Vta -15': 0,
                'M_Vta -15 AA': 1  # Evitar divisiones por 0
            }, inplace=True)

            # Calcular el stock total
            self.df['STOCK_TOTAL'] = self.df['Disponible'] + self.df['Calidad'] + self.df['Stock Externo']

            # Calcular demanda estimada
            self._calculate_demand_estimation()

        except KeyError as e:
            print(f"Error en las columnas del DataFrame: {e}")
            raise

    def _calculate_demand_estimation(self):
        """
        Calcular la estimación de demanda basada en datos históricos
        """
        try:
            # Demanda base y variación anual
            self.df['DEMANDA_MEDIA'] = self.df['M_Vta -15']
            self.df['VAR_ANUAL'] = abs(1 - (self.df['M_Vta -15'] / self.df['M_Vta -15 AA'])) * 100

            # Ajustar demanda para variaciones significativas (>20%)
            mask = self.df['VAR_ANUAL'] > 20
            self.df.loc[mask, 'DEMANDA_MEDIA'] *= (1 + self.df.loc[mask, 'VAR_ANUAL'] / 100)
        except Exception as e:
            print(f"Error al calcular la estimación de demanda: {e}")
            raise

    def optimize_production(self, available_hours: float, maintenance_hours: float, pending_orders: Dict[str, float] = None) -> pd.DataFrame:
        """
        Optimizar la producción usando programación lineal

        :param available_hours: Horas totales disponibles para producción
        :param maintenance_hours: Horas programadas para mantenimiento
        :param pending_orders: Pedidos pendientes por producto
        :return: Plan de producción optimizado
        """
        try:
            # Calcular horas disponibles netas
            net_hours = available_hours - maintenance_hours

            # Parámetros básicos
            production_rates = self.df['Cj/H'].values
            current_stock = self.df['STOCK_TOTAL'].values
            demand = self.df['DEMANDA_MEDIA'].values

            # Validaciones previas
            if net_hours <= 0:
                raise ValueError("No hay horas disponibles para producción.")

            if (current_stock < demand).all():
                print("Advertencia: Stock inicial insuficiente para satisfacer la demanda.")

            # Diagnóstico
            print("Stock inicial:", current_stock)
            print("Demanda estimada:", demand)
            print("Tasas de producción:", production_rates)
            print("Horas disponibles netas:", net_hours)

            # Definir función objetivo: maximizar producción efectiva
            c = -1 * np.minimum(demand, current_stock)

            # Restricciones
            A_ub = []
            b_ub = []

            # Restricción 1: Horas totales disponibles
            A_ub.append(1 / production_rates)
            b_ub.append(net_hours)

            # Restricción 2: Stock de seguridad
            safety_stock = np.minimum(demand * self.SAFETY_STOCK_DAYS, current_stock)
            for i in range(len(production_rates)):
                row = np.zeros(len(production_rates))
                row[i] = -1
                A_ub.append(row)
                b_ub.append(-(safety_stock[i] - current_stock[i]))

            # Restricción 3: Pedidos pendientes
            if pending_orders:
                for code, quantity in pending_orders.items():
                    idx = self.df.index[self.df['COD_ART'] == code]
                    if len(idx) > 0:
                        current_stock[idx[0]] -= quantity

            # Convertir restricciones a arrays numpy
            A_ub = np.array(A_ub, dtype=object)
            b_ub = np.array(b_ub)

            # Resolver el problema de optimización
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')

            if not result.success:
                print("Error en la optimización:", result.message)
                raise ValueError("Optimización no exitosa")

            # Crear plan de producción
            production_plan = self.df.copy()
            production_plan['CAJAS_PRODUCIR'] = result.x
            production_plan['HORAS_PRODUCCION'] = result.x / production_plan['Cj/H']

            return production_plan

        except Exception as e:
            print(f"Error al optimizar la producción: {e}")
            raise

    def generate_production_report(self, production_plan: pd.DataFrame) -> Dict[str, Any]:
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

            # Exportar el informe
            production_plan.to_csv("plan_produccion.csv", index=False, sep=';', encoding='latin1')
            print("Informe de producción generado y guardado en 'plan_produccion.csv'.")

            return report

        except Exception as e:
            print(f"Error al generar el informe de producción: {e}")
            raise

if __name__ == "__main__":
    data_path = "stock_data.csv"

    try:
        # Instanciar el sistema
        system = StockManagementSystem(data_path)

        # Definir parámetros
        available_hours = 100
        maintenance_hours = 5
        pending_orders = {"244719": 50, "274756": 30}

        # Optimizar producción
        production_plan = system.optimize_production(available_hours, maintenance_hours, pending_orders)

        # Generar reporte
        report = system.generate_production_report(production_plan)

        # Mostrar reporte
        print(report)

    except Exception as e:
        print(f"Error durante la ejecución: {e}")
