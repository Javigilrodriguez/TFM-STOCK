import pandas as pd
import numpy as np
from scipy.optimize import linprog
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplexOptimizer:
    def __init__(self):
        self.safety_stock_days = 3
        self.min_production_hours = 2
        self.available_hours_per_day = 24
        self.product_change_penalty = 0.15  # 15% penalización por cambio de producto
        self.group_change_penalty = 0.30    # 30% penalización por cambio de grupo

    def prepare_data(self, df):
        """Prepara los datos aplicando las transformaciones necesarias"""
        try:
            # Convertir columnas numéricas
            df['cajas_hora'] = pd.to_numeric(df['Cj/H'].str.replace(',', '.'), errors='coerce')
            df['ventas_15d'] = pd.to_numeric(df['M_Vta -15'].str.replace(',', '.'), errors='coerce')
            
            # Filtrar productos duplicados manteniendo el código más reciente
            logger.info("Procesando duplicados...")
            df_sorted = df.sort_values('COD_ART', ascending=True)  # Código más reciente al final
            df_unique = df_sorted.drop_duplicates(subset=['NOM_ART'], keep='last')
            
            # Filtrar productos sin ventas (posiblemente descatalogados)
            valid_mask = (df_unique['cajas_hora'].notna()) & (df_unique['ventas_15d'].notna()) & (df_unique['ventas_15d'] > 0)
            valid_products = df_unique[valid_mask].copy()
            
            # Calcular métricas
            valid_products['demanda_diaria'] = valid_products['ventas_15d'] / 15
            valid_products['stock_seguridad'] = valid_products['demanda_diaria'] * self.safety_stock_days
            
            logger.info(f"Total productos originales: {len(df)}")
            logger.info(f"Productos únicos después de filtrar duplicados: {len(df_unique)}")
            logger.info(f"Productos válidos con ventas: {len(valid_products)}")
            
            return valid_products
            
        except Exception as e:
            logger.error(f"Error en preparación de datos: {str(e)}")
            return None

    def build_constraints(self, data, planning_days=7):
        """Construye restricciones asegurando uso mínimo de capacidad"""
        try:
            n_products = len(data)
            total_hours = self.available_hours_per_day * planning_days
            min_total_hours = total_hours * 0.80  # Mínimo 80% de utilización
            
            # 1. Restricción de horas totales (mínimo y máximo)
            A_hours = np.vstack([
                np.ones(n_products),   # Para máximo
                -np.ones(n_products)   # Para mínimo
            ])
            b_hours = np.array([
                total_hours * 0.95,    # Máximo 95% del tiempo
                -min_total_hours       # Mínimo 80% del tiempo
            ])
            
            # 2. Restricciones por producto
            A_products = []
            b_products = []
            
            for i in range(n_products):
                # Mínimo de producción basado en demanda
                min_prod = max(
                    self.min_production_hours,
                    (data.iloc[i]['demanda_diaria'] * 3) / data.iloc[i]['cajas_hora']
                )
                
                # Máximo de producción
                max_prod = min(
                    self.available_hours_per_day,
                    (data.iloc[i]['demanda_diaria'] * 7) / data.iloc[i]['cajas_hora']
                )
                
                # Crear restricciones para este producto
                product_min = np.zeros(n_products)
                product_min[i] = -1
                A_products.append(product_min)
                b_products.append(-min_prod)
                
                product_max = np.zeros(n_products)
                product_max[i] = 1
                A_products.append(product_max)
                b_products.append(max_prod)
            
            # 3. Restricción de balance entre productos
            # Asegurar que productos con más demanda reciban más tiempo
            weights = data['demanda_diaria'].values / data['demanda_diaria'].sum()
            A_balance = np.diag(weights) - np.ones((n_products, n_products)) * weights.reshape(-1, 1)
            b_balance = np.zeros(n_products)
            
            # Combinar todas las restricciones
            A = np.vstack([
                A_hours,
                A_products,
                A_balance
            ])
            
            b = np.concatenate([
                b_hours,
                b_products,
                b_balance
            ])
            
            logger.info(f"Restricciones generadas: {len(b)} restricciones para {n_products} productos")
            return A, b
            
        except Exception as e:
            logger.error(f"Error construyendo restricciones: {str(e)}")
            return None, None
            
            return A, b
            
            return A, b
            
        except Exception as e:
            logger.error(f"Error construyendo restricciones: {str(e)}")
            return None, None

    def optimize(self, df, planning_days=7):
        """Ejecuta la optimización usando el método Simplex"""
        try:
            # Preparar datos
            data = self.prepare_data(df)
            if data is None or len(data) == 0:
                return None
                
            # Construir restricciones
            A, b = self.build_constraints(data, planning_days)
            if A is None or b is None:
                return None
                
            # Función objetivo: maximizar producción priorizada
            priority = (data['demanda_diaria'] * planning_days - data['stock_seguridad']) / data['cajas_hora']
            c = -priority.values  # Negativo porque linprog minimiza
            
            # Resolver usando el método Simplex con parámetros ajustados de HiGHS
            result = linprog(
                c,
                A_ub=A,
                b_ub=b,
                method='highs',
                options={
                    'disp': True,
                    'presolve': False,  # Desactivar presolve para forzar más iteraciones
                    'dual_feasibility_tolerance': 1e-8,  # Tolerancia más estricta
                    'primal_feasibility_tolerance': 1e-8,
                    'time_limit': 10,  # Dar más tiempo para encontrar solución
                    'threads': 1  # Forzar ejecución serial
                }
            )
            
            if result.success:
                # Crear plan de producción
                production_plan = pd.DataFrame({
                    'codigo': data['COD_ART'],
                    'nombre': data['NOM_ART'],
                    'horas_asignadas': result.x,
                    'cajas_producir': result.x * data['cajas_hora'],
                    'demanda_diaria': data['demanda_diaria'],
                    'stock_seguridad': data['stock_seguridad']
                })
                
                # Filtrar y ordenar resultados
                production_plan = production_plan[production_plan['horas_asignadas'] > 0.1]
                production_plan = production_plan.sort_values('horas_asignadas', ascending=False)
                
                logger.info(f"Plan generado para {len(production_plan)} productos")
                return production_plan
                
            else:
                logger.error(f"Optimización falló: {result.message}")
                return None
                
        except Exception as e:
            logger.error(f"Error en optimización: {str(e)}")
            return None

def load_dataset():
    """Busca y carga el primer archivo CSV encontrado en la carpeta Dataset"""
    try:
        # Buscar archivos CSV en la carpeta Dataset
        for file in os.listdir("Dataset"):
            if file.endswith('.csv'):
                file_path = os.path.join("Dataset", file)
                logger.info(f"Cargando archivo: {file}")
                # Cargar el CSV con los parámetros específicos
                return pd.read_csv(file_path, sep=';', encoding='latin1', skiprows=4)
        raise FileNotFoundError("No se encontró ningún archivo CSV en la carpeta Dataset")
    except Exception as e:
        logger.error(f"Error cargando dataset: {str(e)}")
        return None

def main():
    try:
        # Cargar datos del primer CSV encontrado en la carpeta Dataset
        df = load_dataset()
        if df is None:
            return
            
        # Crear y ejecutar optimizador
        optimizer = SimplexOptimizer()
        plan = optimizer.optimize(df)
        
        if plan is not None:
            # Mostrar resumen
            print("\nResumen del plan de producción:")
            print(f"Total horas asignadas: {plan['horas_asignadas'].sum():.2f}")
            print(f"Total productos planificados: {len(plan)}")
            print(f"Total cajas a producir: {plan['cajas_producir'].sum():.0f}")
            
            # Guardar resultados
            plan.to_csv('plan_produccion.csv', index=False)
            print("\nPlan guardado en 'plan_produccion.csv'")
            
    except Exception as e:
        print(f"Error en ejecución: {str(e)}")

if __name__ == "__main__":
    main()