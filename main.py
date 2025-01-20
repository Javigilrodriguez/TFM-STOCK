import pandas as pd
import numpy as np
import os
import logging
from scipy.optimize import linprog

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplexOptimizer:
    def __init__(self):
        self.available_hours = 24 * 2  # 48 horas
        self.min_hours = 1  
        self.max_hours = 12  

    def optimize_production(self, products_df):
        try:
            # Agrupar por código de producto, agregando ventas y tomando la primera tasa de producción
            grouped_df = products_df.groupby('COD_ART').agg({
                'NOM_ART': 'first',  # Tomar el primer nombre
                'Cj/H': 'first',     # Tomar la primera tasa de producción
                'M_Vta -15': 'sum'   # Sumar ventas
            }).reset_index()
            
            # Convertir columnas numéricas
            grouped_df['cajas_hora'] = pd.to_numeric(grouped_df['Cj/H'].str.replace(',', '.'), errors='coerce')
            grouped_df['venta_15d'] = pd.to_numeric(grouped_df['M_Vta -15'].str.replace(',', '.'), errors='coerce')
            
            # Filtrar productos con ventas significativas
            mask = grouped_df['venta_15d'] > 0
            valid_products = grouped_df[mask].copy()
            
            n_products = len(valid_products)
            logger.info(f"Optimizando para {n_products} productos con ventas")
            
            # Calcular demanda diaria y tasas de producción
            daily_demand = valid_products['venta_15d'].values / 15
            production_rates = valid_products['cajas_hora'].values
            
            # Priorizar productos con mayor demanda
            priority = daily_demand / production_rates
            c = -priority  # Negativo para maximizar
            
            # Restricciones más flexibles
            A_ub = np.vstack([
                np.ones((1, n_products)),  # Límite total de horas
                np.eye(n_products),         # Límite superior por producto
                -np.eye(n_products)         # Límite inferior por producto
            ])
            
            b_ub = np.concatenate([
                [self.available_hours],     # Horas totales disponibles
                np.full(n_products, self.max_hours),   # Máximo por producto
                np.full(n_products, -self.min_hours)   # Mínimo por producto
            ])
            
            # Resolver con más opciones
            result = linprog(
                c, 
                A_ub=A_ub, 
                b_ub=b_ub, 
                method='highs',
                options={
                    'disp': False,
                    'presolve': True,
                    'dual_feasibility_tolerance': 1e-4,
                    'primal_feasibility_tolerance': 1e-4
                }
            )
            
            if result.success:
                logger.info("Optimización exitosa")
                hours_assigned = result.x
                production_quantities = hours_assigned * production_rates
                
                # Crear DataFrame de resultados
                results = pd.DataFrame({
                    'codigo': valid_products['COD_ART'],
                    'nombre': valid_products['NOM_ART'],
                    'cajas_hora': valid_products['cajas_hora'],
                    'venta_15d': valid_products['venta_15d'],
                    'produccion_propuesta': production_quantities,
                    'horas_produccion': hours_assigned
                })
                
                # Guardar resultados
                output_dir = "output"
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, "plan_produccion.xlsx")
                results.to_excel(output_file, index=False)
                
                # Mostrar resumen
                print(f"\nResumen del plan:")
                print(f"Total horas planificadas: {results['horas_produccion'].sum():.1f}")
                print(f"Total productos planificados: {len(results)}")
                print(f"Plan guardado en: {output_file}")
                
                return results
            else:
                logger.error(f"Optimización falló: {result.message}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error en optimización: {str(e)}")
            return pd.DataFrame()

def load_first_csv():
    """Carga el primer archivo CSV que encuentre en la carpeta Dataset"""
    dataset_dir = "Dataset"
    for archivo in os.listdir(dataset_dir):
        if archivo.endswith('.csv'):
            file_path = os.path.join(dataset_dir, archivo)
            print(f"Cargando archivo: {archivo}")
            return pd.read_csv(file_path, sep=';', encoding='latin1', skiprows=4)
    raise FileNotFoundError("No se encontró ningún archivo CSV en la carpeta Dataset")

def main():
    try:
        # Cargar datos
        df = load_first_csv()
        
        # Imprimir productos únicos
        unique_products = df.drop_duplicates(subset=['COD_ART'])
        print("Productos únicos en el dataset:")
        for i, (codigo, nombre, cajas_hora, venta_15d) in enumerate(zip(
            unique_products['COD_ART'], 
            unique_products['NOM_ART'], 
            unique_products['Cj/H'], 
            unique_products['M_Vta -15']
        ), 1):
            print(f"{i}. {codigo} - {nombre} (Cajas/Hora: {cajas_hora}, Ventas 15d: {venta_15d})")
        
        # Optimizar
        optimizer = SimplexOptimizer()
        optimizer.optimize_production(df)
        
    except Exception as e:
        print(f"Error en ejecución: {str(e)}")

if __name__ == "__main__":
    main()