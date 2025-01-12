# production_optimizer.py
import numpy as np
import pandas as pd
from config import CHANGEOVER_ARTICLE_PENALTY, CHANGEOVER_GROUP_PENALTY

class ProductionOptimizer:
    """Optimizador de producción usando método Simplex"""
    
    def __init__(self, safety_stock_days=3, min_production_hours=2):
        self.SAFETY_STOCK_DAYS = safety_stock_days
        self.MIN_PRODUCTION_HOURS = min_production_hours
        
    def optimize(self, df, available_hours, maintenance_hours):
        """Optimiza la producción según restricciones"""
        net_hours = available_hours - maintenance_hours
        
        if net_hours <= 0:
            raise ValueError("No hay horas disponibles para producción")
            
        production_rates = np.maximum(df['Cj/H'].values, 1)
        current_stock = df['STOCK_TOTAL'].values
        demand = df['PREDICTED_DEMAND'].values
        
        # Limpiar valores
        demand = np.nan_to_num(demand, nan=0.0, posinf=0.0, neginf=0.0)
        current_stock = np.nan_to_num(current_stock, nan=0.0)
        
        # Diagnóstico inicial
        print("\nDiagnóstico de optimización:")
        print(f"Horas netas disponibles: {net_hours}")
        print(f"Total productos: {len(production_rates)}")
        print(f"Demanda total: {demand.sum():.2f}")
        print(f"Stock total actual: {current_stock.sum():.2f}")
        
        # Calcular prioridades
        priority_score = (demand * self.SAFETY_STOCK_DAYS - current_stock) / production_rates
        priority_order = np.argsort(-priority_score)
        
        # Asignar producción
        production_hours = np.zeros_like(demand)
        hours_used = 0
        products_scheduled = 0
        
        for idx in priority_order:
            if hours_used >= net_hours:
                break
                
            target_stock = demand[idx] * self.SAFETY_STOCK_DAYS
            current = current_stock[idx]
            
            if current < target_stock:
                hours_needed = (target_stock - current) / production_rates[idx]
                
                # Aplicar restricción de horas mínimas
                if hours_needed < self.MIN_PRODUCTION_HOURS:
                    if hours_used + self.MIN_PRODUCTION_HOURS <= net_hours:
                        hours_needed = self.MIN_PRODUCTION_HOURS
                    else:
                        continue
                
                # Asignar horas disponibles
                hours_to_assign = min(hours_needed, net_hours - hours_used)
                if hours_to_assign >= self.MIN_PRODUCTION_HOURS:
                    production_hours[idx] = hours_to_assign
                    hours_used += hours_to_assign
                    products_scheduled += 1
        
        # Ajustar por cambios de producto
        if products_scheduled > 1:
            available_time_reduction = (products_scheduled - 1) * CHANGEOVER_ARTICLE_PENALTY
            net_hours = net_hours * (1 - available_time_reduction)
            
            # Reajustar si es necesario
            if hours_used > net_hours:
                scale_factor = net_hours / hours_used
                production_hours *= scale_factor
                hours_used = net_hours
        
        # Calcular cantidades finales
        production_quantities = production_hours * production_rates
        
        print("\nResumen de la solución:")
        print(f"Total cajas a producir: {production_quantities.sum():.2f}")
        print(f"Total horas asignadas: {hours_used:.2f}")
        print(f"Productos programados: {products_scheduled}")
        
        return production_hours, production_quantities, hours_used