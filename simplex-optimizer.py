import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionPlanner:
    def __init__(self, safety_stock_days=3, min_production_hours=2):
        self.safety_stock_days = safety_stock_days
        self.min_production_hours = min_production_hours

    def load_data(self, folder_path="Dataset"):
        try:
            all_data = []
            for file in os.listdir(folder_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(folder_path, file)
                    logger.info(f"Cargando archivo: {file}")
                    
                    # Leer la fecha del archivo
                    with open(file_path, 'r', encoding='latin1') as f:
                        date_str = f.readline().split(';')[1].strip().split()[0]
                        logger.info(f"Fecha del archivo: {date_str}")
                    
                    # Leer el CSV
                    df = pd.read_csv(file_path, sep=';', encoding='latin1', skiprows=4)
                    df = df[df['COD_ART'].notna()]  # Eliminar filas sin código
                    all_data.append(df)
            
            df = pd.concat(all_data, ignore_index=True)
            
            # Convertir columnas numéricas
            for col in ['Cj/H', 'M_Vta -15', 'Disponible', 'Calidad', 'Stock Externo']:
                df[col] = pd.to_numeric(df[col].replace({'(en blanco)': '0', ',': '.'}, regex=True), errors='coerce')
            
            # Eliminar duplicados
            df = df.drop_duplicates(subset=['COD_ART'], keep='last')
            logger.info(f"Datos cargados: {len(df)} productos únicos")
            
            return df
            
        except Exception as e:
            logger.error(f"Error cargando datos: {str(e)}")
            return None

    def generate_production_plan(self, df, available_hours, planning_days=7):
        try:
            # Copiar solo las columnas necesarias
            plan = df.copy()[['COD_ART', 'NOM_ART', 'Cj/H', 'M_Vta -15', 'Disponible', 'Calidad', 'Stock Externo']]
            
            # Calcular campos básicos
            plan['cajas_hora'] = plan['Cj/H']
            plan['stock_total'] = plan['Disponible'].fillna(0) + plan['Calidad'].fillna(0) + plan['Stock Externo'].fillna(0)
            plan['demanda_diaria'] = plan['M_Vta -15'].fillna(0) / 15
            
            # Filtrar productos válidos
            plan = plan[plan['cajas_hora'] > 0].copy()
            logger.info(f"Planificando producción para {len(plan)} productos")
            
            # Calcular necesidades
            plan['cajas_necesarias'] = np.maximum(
                0, 
                (self.safety_stock_days * plan['demanda_diaria']) - plan['stock_total']
            )
            
            # Aplicar lote mínimo
            plan['cajas_a_producir'] = np.maximum(
                plan['cajas_necesarias'],
                self.min_production_hours * plan['cajas_hora']
            )
            
            # Calcular horas
            plan['horas_necesarias'] = plan['cajas_a_producir'] / plan['cajas_hora']
            
            # Filtrar y ordenar
            plan = plan[plan['cajas_a_producir'] > 0].sort_values('stock_total')
            
            # Ajustar al tiempo disponible
            horas_totales = 0
            productos_final = []
            
            for _, row in plan.iterrows():
                if horas_totales + row['horas_necesarias'] <= available_hours:
                    productos_final.append(row)
                    horas_totales += row['horas_necesarias']
            
            plan_final = pd.DataFrame(productos_final)
            
            if not plan_final.empty:
                fecha_inicio = datetime.now()
                fecha_fin = fecha_inicio + timedelta(days=planning_days)
                
                # Guardar plan con fechas
                filename = f"plan_produccion_{fecha_inicio.strftime('%Y%m%d')}_{fecha_fin.strftime('%Y%m%d')}.csv"
                
                # Crear encabezado con fechas
                with open(filename, 'w', encoding='latin1') as f:
                    f.write(f"Fecha Inicio;{fecha_inicio.strftime('%d/%m/%Y')};;;;;;;;\n")
                    f.write(f"Fecha Fin;{fecha_fin.strftime('%d/%m/%Y')};;;;;;;;\n")
                    f.write(";;;;;;;;;;;\n")
                    f.write(";;;;;;;;;;;\n")
                
                # Guardar el plan después del encabezado
                plan_final.to_csv(filename, sep=';', index=False, mode='a', encoding='latin1')
                logger.info(f"Plan guardado en: {filename}")
                
                return plan_final, fecha_inicio, fecha_fin
            
            return None, None, None
            
        except Exception as e:
            logger.error(f"Error en plan de producción: {str(e)}")
            return None, None, None

    def generate_report(self, plan, fecha_inicio, fecha_fin):
        if plan is None or len(plan) == 0:
            logger.error("No hay plan de producción para reportar")
            return
            
        print(f"\nPLAN DE PRODUCCIÓN ({fecha_inicio.strftime('%d/%m/%Y')} - {fecha_fin.strftime('%d/%m/%Y')})")
        print("-" * 80)
        print(f"Total productos: {len(plan)}")
        print(f"Total cajas: {plan['cajas_a_producir'].sum():.0f}")
        print(f"Total horas: {plan['horas_necesarias'].sum():.1f}")
        print("\nDetalle por producto:")
        print(plan[['COD_ART', 'NOM_ART', 'stock_total', 'cajas_a_producir', 'horas_necesarias']].to_string(index=False))

def main():
    try:
        planner = ProductionPlanner()
        df = planner.load_data()
        if df is not None:
            plan, fecha_inicio, fecha_fin = planner.generate_production_plan(df, available_hours=24*7)
            planner.generate_report(plan, fecha_inicio, fecha_fin)
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()