import numpy as np
import pandas as pd
from scipy.optimize import linprog
from typing import Dict, List, Tuple, Any

class StockManagementSystem:
    def __init__(self, data_path: str):
        """
        Inicializar el sistema de gestión de inventario con los datos de entrada
        
        :param data_path: Ruta al archivo CSV que contiene los datos de inventario
        """
        try:
            # Leer el archivo CSV con el separador adecuado y codificación latin1
            self.df = pd.read_csv(data_path, sep=';', encoding='latin1', skiprows=3, header=0)
            # Renombrar columnas para corregir encabezados mal formateados
            self.df.columns = self.df.iloc[0]  # Usar la primera fila como encabezado
            self.df = self.df[1:]  # Eliminar la fila que se usó como encabezado
            self.df.columns = self.df.columns.str.strip()  # Quitar espacios en blanco

            print("Columnas cargadas y corregidas:", self.df.columns)
        except Exception as e:
            print(f"Error al cargar el archivo CSV: {e}")
            raise
        
        # Limpiar el DataFrame
        self._clean_data()
        
        # Parámetros del sistema
        self.SAFETY_STOCK_DAYS = 3
        self.MIN_PRODUCTION_HOURS = 2
        self.PRODUCTION_CHANGE_PENALTY_ART = 0.15  # Penalización del 15% por cambio de producto
        self.PRODUCTION_CHANGE_PENALTY_GROUP = 0.30  # Penalización del 30% por cambio de grupo
        
    def _clean_data(self):
        """
        Limpiar y preparar los datos para su análisis
        """
        try:
            # Eliminación de filas completamente vacías y reseteo de índices
            self.df = self.df.dropna(how='all').reset_index(drop=True)
            self.df = self.df[self.df['COD_ART'] != 'Total general']

            # Convertir columnas relevantes a valores numéricos
            num_columns = ['Cj/H', 'Disponible', 'Calidad', 'Stock Externo', 
                        'M_Vta -15', 'M_Vta -15 AA']
            for col in num_columns:
                self.df[col] = self.df[col].replace({',': '.'}, regex=True)  # Reemplazar comas por puntos decimales
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

            # Reemplazar valores nulos en las columnas clave con valores predeterminados
            self.df['Cj/H'] = self.df['Cj/H'].fillna(1)  # Producción mínima de 1 caja/hora si está vacío
            self.df['Disponible'] = self.df['Disponible'].fillna(0)
            self.df['Calidad'] = self.df['Calidad'].fillna(0)
            self.df['Stock Externo'] = self.df['Stock Externo'].fillna(0)
            self.df['M_Vta -15'] = self.df['M_Vta -15'].fillna(0)
            self.df['M_Vta -15 AA'] = self.df['M_Vta -15 AA'].fillna(1)  # Evitar divisiones por 0

            # Calcular el stock total
            self.df['STOCK_TOTAL'] = (
                self.df['Disponible'] + 
                self.df['Calidad'] + 
                self.df['Stock Externo']
            ).fillna(0)

            # Calcular la estimación de demanda basada en el histórico
            self._calculate_demand_estimation()

            # Verificar y reemplazar nulos adicionales después de todos los cálculos
            self.df.fillna(0, inplace=True)
        except KeyError as e:
            print(f"Error en las columnas del DataFrame: {e}")
            raise
        except Exception as e:
            print(f"Error al limpiar los datos: {e}")
            raise


    def _calculate_demand_estimation(self):
        """
        Calcular la estimación de demanda considerando datos históricos y tendencias
        """
        try:
            # Demanda base a partir de las ventas recientes
            self.df['DEMANDA_MEDIA'] = self.df['M_Vta -15'].fillna(0)

            # Calcular la variación interanual
            self.df['VAR_ANUAL'] = abs(1 - (
                self.df['M_Vta -15'].fillna(0) / 
                self.df['M_Vta -15 AA'].fillna(1)
            )) * 100

            # Ajustar la tendencia para variaciones significativas (>20%)
            mask = self.df['VAR_ANUAL'] > 20
            self.df.loc[mask, 'DEMANDA_MEDIA'] *= (1 + self.df.loc[mask, 'VAR_ANUAL'] / 100)
        except KeyError as e:
            print(f"Error en las columnas necesarias para la estimación de demanda: {e}")
            raise
        except Exception as e:
            print(f"Error al calcular la estimación de demanda: {e}")
            raise

    def optimize_production(
    self, 
    available_hours: float, 
    maintenance_hours: float,
    pending_orders: Dict[str, float] = None
) -> pd.DataFrame:
        """
        Optimizar la producción usando programación lineal con restricciones mejoradas
        
        :param available_hours: Horas totales disponibles para producción
        :param maintenance_hours: Horas programadas para mantenimiento
        :param pending_orders: Diccionario de pedidos pendientes por código de producto
        :return: Plan de producción optimizado
        """
        try:
            # Asegurarnos de que las columnas clave no tengan valores nulos
            self.df['Cj/H'] = self.df['Cj/H'].fillna(1)
            self.df['STOCK_TOTAL'] = self.df['STOCK_TOTAL'].fillna(0)
            self.df['DEMANDA_MEDIA'] = self.df['DEMANDA_MEDIA'].fillna(0)

            # Filtrar filas con valores inválidos para evitar errores
            self.df = self.df[(self.df['Cj/H'] > 0) & (self.df['Cj/H'].notna())]

            # Ajustar las horas disponibles para el mantenimiento
            net_hours = available_hours - maintenance_hours

            # Parámetros básicos
            production_rates = self.df['Cj/H'].values
            current_stock = self.df['STOCK_TOTAL'].values
            demand = self.df['DEMANDA_MEDIA'].values

            # Objetivo: Minimizar el déficit de inventario
            c = -1 * np.maximum(demand - current_stock, 0)  # Garantizar que no haya valores negativos

            # Matriz de restricciones
            A_ub = []
            b_ub = []

            # 1. Restricción de horas totales de producción
            A_ub.append(1 / np.where(production_rates == 0, 1, production_rates))  # Evitar divisiones por 0
            b_ub.append(net_hours)

            # 2. Restricción de horas mínimas de producción
            min_production = self.MIN_PRODUCTION_HOURS * production_rates
            A_min = -1 * np.eye(len(production_rates))
            b_min = -1 * min_production
            A_ub.extend(A_min)
            b_ub.extend(b_min)

            # 3. Restricción de stock de seguridad
            safety_stock = demand * self.SAFETY_STOCK_DAYS
            A_safety = -1 * np.eye(len(production_rates))
            b_safety = -1 * (safety_stock - current_stock)
            A_ub.extend(A_safety)
            b_ub.extend(b_safety)

            # 4. Restricción de pedidos pendientes (si se proporcionan)
            if pending_orders:
                pending_array = np.zeros(len(production_rates))
                for code, quantity in pending_orders.items():
                    idx = self.df[self.df['COD_ART'] == code].index
                    if len(idx) > 0:
                        pending_array[idx[0]] = quantity
                A_pending = -1 * np.eye(len(production_rates))
                b_pending = -1 * (pending_array - current_stock)
                A_ub.extend(A_pending)
                b_ub.extend(b_pending)

            # Resolver el problema de optimización
            result = linprog(
                c, 
                A_ub=np.array(A_ub), 
                b_ub=np.array(b_ub),
                bounds=(0, None),
                method='highs'
            )

            # Crear un DataFrame con el plan de producción
            production_plan = self.df.copy()
            production_plan['CAJAS_PRODUCIR'] = result.x
            production_plan['HORAS_PRODUCCION'] = np.divide(result.x, production_plan['Cj/H'], 
                                                            out=np.zeros_like(result.x), 
                                                            where=production_plan['Cj/H'] > 0)

            # Calcular las penalizaciones por cambios de producción
            production_plan['CAMBIOS_PRODUCTO'] = (production_plan['CAJAS_PRODUCIR'] > 0).astype(int)
            production_plan['PENALIZACION_CAMBIO'] = (
                production_plan['CAMBIOS_PRODUCTO'] * self.PRODUCTION_CHANGE_PENALTY_ART +
                (production_plan['CAMBIOS_PRODUCTO'].diff().fillna(0) != 0) * self.PRODUCTION_CHANGE_PENALTY_GROUP
            )

            return production_plan
        except Exception as e:
            print(f"Error al optimizar la producción: {e}")
            raise

    def generate_production_report(
        self, 
        production_plan: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generar un informe completo de la producción
        
        :param production_plan: Plan de producción optimizado
        :return: Resumen del informe de producción
        """
        try:
            report = {
                'total_production': production_plan['CAJAS_PRODUCIR'].sum(),
                                'total_hours': production_plan['HORAS_PRODUCCION'].sum(),
                'product_changes': production_plan['CAMBIOS_PRODUCTO'].sum(),
                'total_change_penalty': production_plan['PENALIZACION_CAMBIO'].sum(),
                'products_to_produce': production_plan[
                    production_plan['CAJAS_PRODUCIR'] > 0
                ][['COD_ART', 'NOM_ART', 'CAJAS_PRODUCIR', 'HORAS_PRODUCCION']].to_dict('records'),
                'potential_stockouts': production_plan[
                    production_plan['STOCK_TOTAL'] < 
                    production_plan['DEMANDA_MEDIA'] * self.SAFETY_STOCK_DAYS
                ][['COD_ART', 'NOM_ART', 'STOCK_TOTAL', 'DEMANDA_MEDIA']].to_dict('records')
            }

            # Mostrar un resumen de los datos principales para validación
            print("Informe de Producción Generado:\n", report)

            # Exportar el informe a un archivo CSV
            production_plan.to_csv("plan_produccion.csv", index=False, sep=';', encoding='latin1')
            print("El plan de producción ha sido guardado en 'plan_produccion.csv'.")

            return report
        except Exception as e:
            print(f"Error al generar el informe de producción: {e}")
            raise

    def display_summary_table(self, production_plan: pd.DataFrame):
        """
        Mostrar una tabla resumen con los principales datos del plan de producción
        
        :param production_plan: Plan de producción optimizado
        """
        try:
            summary_table = production_plan[[
                'COD_ART', 'NOM_ART', 'CAJAS_PRODUCIR', 'HORAS_PRODUCCION', 'STOCK_TOTAL'
            ]].head(10)  # Mostrar las primeras 10 filas como ejemplo
            print("\nTabla Resumen del Plan de Producción:\n")
            print(summary_table)
        except Exception as e:
            print(f"Error al mostrar la tabla resumen: {e}")
            raise


if __name__ == "__main__":
    # Ruta al archivo CSV en la raíz del proyecto
    data_path = "stock_data.csv"

    try:
        # Crear una instancia del sistema
        system = StockManagementSystem(data_path)

        # Definir parámetros de prueba
        available_hours = 100  # Horas totales disponibles para producción
        maintenance_hours = 5  # Horas destinadas al mantenimiento
        pending_orders = {
            "244719": 50,  # Código de producto con cantidad pendiente
            "274756": 30
        }

        # Optimizar la producción
        production_plan = system.optimize_production(available_hours, maintenance_hours, pending_orders)

        # Generar y mostrar el informe de producción
        report = system.generate_production_report(production_plan)

        # Mostrar tabla resumen
        system.display_summary_table(production_plan)

    except Exception as e:
        print(f"Error durante la ejecución: {e}")

