import pandas as pd
import numpy as np
from scipy.optimize import linprog
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductionOptimizer:
    def __init__(self, safety_stock_days=3):
        self.safety_stock_days = safety_stock_days
        self.available_hours = 24  # Horas por día
        self.planning_days = 7     # Días de planificación
        self.min_production_hours = 2  # Mínimo de horas por lote

    def prepare_data(self, df):
        """Prepara y limpia los datos para la optimización"""
        # Convertir columnas numéricas, reemplazando comas por puntos
        df['cajas_hora'] = pd.to_numeric(df['Cj/H'].str.replace(',', '.'), errors='coerce')
        df['ventas_15d'] = pd.to_numeric(df['M_Vta -15'].str.replace(',', '.'), errors='coerce')
        
        # Filtrar productos con datos válidos
        valid_products = df[
            (df['cajas_hora'].notna()) & 
            (df['ventas_15d'].notna()) & 
            (df['cajas_hora'] > 0)
        ].copy()
        
        # Calcular métricas diarias
        valid_products['demanda_diaria'] = valid_products['ventas_15d'] / 15
        
        return valid_products

    def optimize(self, df):
        """Realiza la optimización de la producción"""
        try:
            # Preparar datos
            data = self.prepare_data(df)
            n_products = len(data)
            
            if n_products == 0:
                logger.error("No hay productos válidos para optimizar")
                return None

            logger.info(f"Optimizando producción para {n_products} productos")

            # Variables para el problema de optimización
            total_hours = self.available_hours * self.planning_days
            
            # Coeficientes de la función objetivo (maximizar producción priorizada)
            priorities = data['demanda_diaria'] / data['cajas_hora']
            c = -priorities.values  # Negativo porque linprog minimiza

            # Restricciones
            # 1. Límite total de horas
            A_hours = np.ones((1, n_products))
            b_hours = np.array([total_hours])

            # 2. Límites por producto
            A_product = np.eye(n_products)
            b_product = np.full(n_products, self.available_hours)  # Máximo por producto

            # 3. Producción mínima
            A_min = -np.eye(n_products)
            b_min = np.full(n_products, -self.min_production_hours)

            # Combinar restricciones
            A = np.vstack([A_hours, A_product, A_min])
            b = np.concatenate([b_hours, b_product, b_min])

            # Resolver
            result = linprog(
                c,
                A_ub=A,
                b_ub=b,
                method='highs',
                options={'disp': True}
            )

            if result.success:
                # Crear DataFrame con resultados
                production_plan = pd.DataFrame({
                    'codigo': data['COD_ART'],
                    'nombre': data['NOM_ART'],
                    'horas_asignadas': result.x,
                    'cajas_producir': result.x * data['cajas_hora'],
                    'demanda_diaria': data['demanda_diaria']
                })
                
                # Ordenar por horas asignadas descendente
                production_plan = production_plan.sort_values('horas_asignadas', ascending=False)
                
                return production_plan
            else:
                logger.error(f"La optimización falló: {result.message}")
                return None

        except Exception as e:
            logger.error(f"Error en la optimización: {str(e)}")
            return None

def load_and_optimize():
    try:
        # Cargar datos
        df = pd.read_csv('Dataset 020125.csv', sep=';', encoding='latin1', skiprows=4)
        
        # Crear y ejecutar optimizador
        optimizer = ProductionOptimizer()
        plan = optimizer.optimize(df)
        
        if plan is not None:
            print("\nPlan de producción generado:")
            print(plan.to_string())
            
            print("\nResumen:")
            print(f"Total horas asignadas: {plan['horas_asignadas'].sum():.2f}")
            print(f"Total productos planificados: {len(plan)}")
            
            # Guardar resultados
            plan.to_csv('plan_produccion.csv', index=False)
            print("\nPlan guardado en 'plan_produccion.csv'")
        
    except Exception as e:
        print(f"Error en la ejecución: {str(e)}")

if __name__ == "__main__":
    load_and_optimize()