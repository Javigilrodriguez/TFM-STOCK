# main.py
from datetime import datetime
from stock_management import StockManagementSystem
import argparse

def parse_arguments():
    """Parsea los argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description='Sistema de Gestión de Stock')
    
    parser.add_argument('--data', type=str, required=True,
                      help='Ruta al archivo de datos de stock')
    
    parser.add_argument('--plan-date', type=str, required=True,
                      help='Fecha de planificación (YYYY-MM-DD)')
    
    parser.add_argument('--expected-results', type=str,
                      help='Ruta al archivo de resultados esperados')
    
    parser.add_argument('--available-hours', type=float, default=100,
                      help='Horas disponibles para producción')
    
    parser.add_argument('--maintenance-hours', type=float, default=5,
                      help='Horas de mantenimiento')
    
    parser.add_argument('--output', type=str, default="plan_produccion.csv",
                      help='Ruta para guardar el plan de producción')
    
    return parser.parse_args()

def main():
    """Función principal del sistema"""
    try:
        print("=" * 50)
        print("SISTEMA DE GESTIÓN DE STOCK")
        print("=" * 50)
        print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Parsear argumentos
        args = parse_arguments()
        
        # Inicializar sistema
        print("\nInicializando sistema...")
        system = StockManagementSystem(
            data_path=args.data,
            plan_date=args.plan_date,
            expected_results_path=args.expected_results
        )

        # Optimizar producción
        print("\nOptimizando producción...")
        production_plan = system.optimize_production(
            args.available_hours,
            args.maintenance_hours
        )

        # Generar reporte
        print("\nGenerando reporte...")
        report = system.generate_production_report(
            production_plan,
            args.output
        )

        print("\nProceso completado exitosamente")
        print(f"Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        raise

if __name__ == "__main__":
    main()