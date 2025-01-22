import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionPlanner:
    def __init__(self, 
                 safety_stock_days=3, 
                 min_production_hours=2,
                 working_hours_per_day=24,
                 fecha_inicio=None,
                 fecha_fin=None):
        self.safety_stock_days = safety_stock_days
        self.min_production_hours = min_production_hours
        self.working_hours_per_day = working_hours_per_day
        self.fecha_inicio = fecha_inicio if fecha_inicio else datetime.now()
        self.fecha_fin = fecha_fin if fecha_fin else self.fecha_inicio + timedelta(days=7)

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

    def generate_production_plan(self, df, available_hours=None, dias_habiles=None):
        try:
            # Si no se especifica available_hours, calcular basado en días hábiles
            if available_hours is None:
                if dias_habiles is None:
                    dias_habiles = (self.fecha_fin - self.fecha_inicio).days
                available_hours = dias_habiles * self.working_hours_per_day
            
            # Copiar solo las columnas necesarias
            plan = df.copy()[['COD_ART', 'NOM_ART', 'Cj/H', 'M_Vta -15', 'Disponible', 'Calidad', 'Stock Externo']]
            
            # Resto del código igual que antes...
            plan['cajas_hora'] = plan['Cj/H']
            plan['stock_total'] = plan['Disponible'].fillna(0) + plan['Calidad'].fillna(0) + plan['Stock Externo'].fillna(0)
            plan['demanda_diaria'] = plan['M_Vta -15'].fillna(0) / 15
            
            plan = plan[plan['cajas_hora'] > 0].copy()
            logger.info(f"Planificando producción para {len(plan)} productos")
            
            plan['cajas_necesarias'] = np.maximum(
                0, 
                (self.safety_stock_days * plan['demanda_diaria']) - plan['stock_total']
            )
            
            plan['min_produccion'] = self.min_production_hours * plan['cajas_hora']
            
            plan['cajas_a_producir'] = np.where(
                plan['cajas_necesarias'] > 0,
                np.ceil(
                    np.maximum(
                        plan['cajas_necesarias'],
                        plan['min_produccion']
                    )
                ),
                0
            )
            
            plan['horas_necesarias'] = plan['cajas_a_producir'] / plan['cajas_hora']
            
            plan = plan[plan['cajas_a_producir'] > 0].sort_values('stock_total')
            
            horas_totales = 0
            productos_final = []
            
            for _, row in plan.iterrows():
                if horas_totales + row['horas_necesarias'] <= available_hours:
                    productos_final.append(row)
                    horas_totales += row['horas_necesarias']
            
            plan_final = pd.DataFrame(productos_final)
            
            if not plan_final.empty:
                # Guardar plan con fechas
                filename = f"plan_produccion_{self.fecha_inicio.strftime('%Y%m%d')}_{self.fecha_fin.strftime('%Y%m%d')}.csv"
                
                plan_final_save = plan_final.copy()
                plan_final_save['cajas_a_producir'] = plan_final_save['cajas_a_producir'].astype(int)
                
                with open(filename, 'w', encoding='latin1') as f:
                    f.write(f"Fecha Inicio;{self.fecha_inicio.strftime('%d/%m/%Y')};;;;;;;;\n")
                    f.write(f"Fecha Fin;{self.fecha_fin.strftime('%d/%m/%Y')};;;;;;;;\n")
                    f.write(";;;;;;;;;;;\n")
                    f.write(";;;;;;;;;;;\n")
                
                plan_final_save.to_csv(filename, sep=';', index=False, mode='a', encoding='latin1')
                logger.info(f"Plan guardado en: {filename}")
                
                return plan_final, self.fecha_inicio, self.fecha_fin
            
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
        print(f"Total cajas: {int(plan['cajas_a_producir'].sum())}")
        print(f"Total horas: {plan['horas_necesarias'].sum():.1f}")
        print("\nDetalle por producto:")
        
        display_plan = plan.copy()
        display_plan['cajas_a_producir'] = display_plan['cajas_a_producir'].astype(int)
        display_plan['horas_necesarias'] = display_plan['horas_necesarias'].round(1)
        
        print(display_plan[['COD_ART', 'NOM_ART', 'stock_total', 'cajas_a_producir', 'horas_necesarias']].to_string(index=False))

def main():
    try:
        # Configuración de parámetros
        params = {
            'safety_stock_days': 3,              # Días de stock de seguridad
            'min_production_hours': 2,           # Horas mínimas de producción por lote
            'working_hours_per_day': 24,         # Horas de trabajo por día
            'fecha_inicio': datetime(2024, 1, 22), # Fecha inicial
            'fecha_fin': datetime(2024, 1, 29),    # Fecha final
            'dias_habiles': 5,                   # Días hábiles para producción
            'available_hours': None              # Horas totales disponibles (si es None, se calcula con días hábiles)
        }
        
        # Inicializar el planificador con los parámetros básicos
        planner = ProductionPlanner(
            safety_stock_days=params['safety_stock_days'],
            min_production_hours=params['min_production_hours'],
            working_hours_per_day=params['working_hours_per_day'],
            fecha_inicio=params['fecha_inicio'],
            fecha_fin=params['fecha_fin']
        )
        
        # Cargar datos
        df = planner.load_data()
        if df is not None:
            # Generar plan con parámetros adicionales
            plan, fecha_inicio, fecha_fin = planner.generate_production_plan(
                df,
                available_hours=params['available_hours'],
                dias_habiles=params['dias_habiles']
            )
            planner.generate_report(plan, fecha_inicio, fecha_fin)
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()