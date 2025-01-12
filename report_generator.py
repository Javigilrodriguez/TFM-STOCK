# report_generator.py
import pandas as pd
from datetime import datetime

class ReportGenerator:
    """Generador de reportes de producción"""
    
    @staticmethod
    def generate_report(production_plan, output_path="plan_produccion.csv"):
        """Genera reporte detallado de producción"""
        try:
            # Calcular métricas generales
            total_production = production_plan['CAJAS_PRODUCIR'].sum()
            total_hours = production_plan['HORAS_PRODUCCION'].sum()
            products_to_produce = len(production_plan[production_plan['CAJAS_PRODUCIR'] > 0])
            
            # Calcular métricas por producto
            products_detail = []
            for _, row in production_plan[production_plan['CAJAS_PRODUCIR'] > 0].iterrows():
                product_detail = {
                    'COD_ART': row['COD_ART'],
                    'CAJAS_PRODUCIR': row['CAJAS_PRODUCIR'],
                    'HORAS_PRODUCCION': row['HORAS_PRODUCCION'],
                    'STOCK_ACTUAL': row['STOCK_TOTAL'],
                    'DEMANDA_PREDICHA': row.get('PREDICTED_DEMAND', 0),
                    'DIAS_COBERTURA': row.get('DIAS_COBERTURA', 0)
                }
                products_detail.append(product_detail)
            
            # Crear reporte completo
            report = {
                'fecha_generacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'metricas_generales': {
                    'total_production': total_production,
                    'total_hours': total_hours,
                    'products_to_produce': products_to_produce,
                    'average_lot_size': total_production / products_to_produce if products_to_produce > 0 else 0,
                },
                'metricas_eficiencia': {
                    'utilizacion_tiempo': (total_hours / 24) * 100 if total_hours <= 24 else 100,
                    'productos_programados': products_to_produce,
                    'cajas_por_hora': total_production / total_hours if total_hours > 0 else 0
                },
                'products_detail': products_detail
            }

            # Guardar plan detallado
            production_plan.to_csv(output_path, index=False, sep=';', encoding='latin1')
            print(f"\nPlan de producción guardado en: {output_path}")

            return report

        except Exception as e:
            print(f"Error generando informe: {e}")
            raise
    
    @staticmethod
    def print_report_summary(report):
        """Imprime un resumen del reporte de producción"""
        try:
            print("\nRESUMEN DEL PLAN DE PRODUCCIÓN")
            print("=" * 40)
            print(f"Fecha generación: {report['fecha_generacion']}")
            
            print("\nMÉTRICAS GENERALES:")
            print(f"Total cajas a producir: {report['metricas_generales']['total_production']:.0f}")
            print(f"Total horas asignadas: {report['metricas_generales']['total_hours']:.2f}")
            print(f"Productos programados: {report['metricas_generales']['products_to_produce']}")
            print(f"Tamaño medio de lote: {report['metricas_generales']['average_lot_size']:.0f}")
            
            print("\nMÉTRICAS DE EFICIENCIA:")
            print(f"Utilización del tiempo: {report['metricas_eficiencia']['utilizacion_tiempo']:.1f}%")
            print(f"Productividad: {report['metricas_eficiencia']['cajas_por_hora']:.0f} cajas/hora")
            
            print("\nDETALLE DE PRODUCTOS:")
            for product in report['products_detail']:
                print(f"\nCódigo: {product['COD_ART']}")
                print(f"  Cajas a producir: {product['CAJAS_PRODUCIR']:.0f}")
                print(f"  Horas asignadas: {product['HORAS_PRODUCCION']:.2f}")
                print(f"  Stock actual: {product['STOCK_ACTUAL']:.0f}")
                print(f"  Demanda predicha: {product['DEMANDA_PREDICHA']:.0f}")
                print(f"  Días cobertura: {product['DIAS_COBERTURA']:.1f}")

        except Exception as e:
            print(f"Error imprimiendo resumen: {e}")
            raise
    
    @staticmethod
    def generate_alerts(production_plan):
        """Genera alertas sobre el plan de producción"""
        alerts = []
        
        # Alerta de productos sin stock suficiente
        low_stock = production_plan[
            (production_plan['STOCK_TOTAL'] < production_plan['PREDICTED_DEMAND']) &
            (production_plan['CAJAS_PRODUCIR'] == 0)
        ]
        
        if not low_stock.empty:
            alerts.append({
                'tipo': 'STOCK_BAJO',
                'mensaje': f"Hay {len(low_stock)} productos con stock insuficiente sin producción programada",
                'productos': low_stock['COD_ART'].tolist()
            })
        
        # Alerta de productos con alta variación
        if 'VAR_ANUAL' in production_plan.columns:
            high_var = production_plan[abs(production_plan['VAR_ANUAL']) > 50]
            if not high_var.empty:
                alerts.append({
                    'tipo': 'VARIACION_ALTA',
                    'mensaje': f"Hay {len(high_var)} productos con variación anual superior al 50%",
                    'productos': high_var['COD_ART'].tolist()
                })
        
        return alerts